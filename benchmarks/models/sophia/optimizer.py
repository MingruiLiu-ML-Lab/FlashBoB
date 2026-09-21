import math

import torch
import torch.distributed as dist


class SophiaH(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 6e-4,
        betas=(0.96, 0.99),
        gamma: float = 0.01,
        weight_decay: float = 0.2,
        eps: float = 1e-12,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid beta2: {betas[1]}")
        if gamma < 0.0:
            raise ValueError(f"invalid gamma: {gamma}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight decay: {weight_decay}")
        if eps <= 0.0:
            raise ValueError(f"invalid eps: {eps}")

        super().__init__(
            params,
            dict(lr=lr, betas=betas, gamma=gamma, weight_decay=weight_decay, eps=eps),
        )

    def _init_state(self, p: torch.Tensor):
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32)
            state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32)
        return state

    @torch.no_grad()
    def update_hessian(self, hessian_estimates):
        flat_idx = 0
        for group in self.param_groups:
            beta2 = group["betas"][1]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self._init_state(p)
                h_est = hessian_estimates[flat_idx].float()
                flat_idx += 1
                state["hessian"].mul_(beta2).add_(h_est, alpha=1.0 - beta2)

        if flat_idx != len(hessian_estimates):
            raise ValueError("mismatched number of hessian estimates")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, _ = group["betas"]
            lr = group["lr"]
            gamma = group["gamma"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self._init_state(p)
                grad = p.grad.float()
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                state["step"] += 1
                step = state["step"]

                p.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                denom = gamma * torch.relu(hessian) + eps
                ratio = (exp_avg.abs() / denom).clamp(max=1.0)
                step_size = lr / (1.0 - beta1**step)
                update = exp_avg.sign() * ratio
                p.add_(update.to(dtype=p.dtype), alpha=-step_size)

    @torch.no_grad()
    def logging_stats(self) -> dict:
        total_params = 0
        clipped = 0
        momentum_sq = 0.0
        hessian_sq = 0.0
        hessian_l1 = 0.0
        update_sq = 0.0

        for group in self.param_groups:
            gamma = group["gamma"]
            eps = group["eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self._init_state(p)
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]
                denom = gamma * torch.relu(hessian) + eps
                ratio = exp_avg.abs() / denom
                clipped += ratio.ge(1.0).sum().item()
                total_params += ratio.numel()
                momentum_sq += exp_avg.pow(2).sum().item()
                hessian_sq += hessian.pow(2).sum().item()
                hessian_l1 += hessian.abs().sum().item()
                update_sq += ratio.clamp(max=1.0).pow(2).sum().item()

        clipped_frac = 0.0 if total_params == 0 else clipped / total_params
        return {
            "optimizer/momentum_l2": math.sqrt(momentum_sq),
            "optimizer/hessian_l2": math.sqrt(hessian_sq),
            "optimizer/hessian_l1": hessian_l1,
            "optimizer/clipped_frac": clipped_frac,
            "optimizer/unclipped_frac": 1.0 - clipped_frac,
            "optimizer/precond_update_l2": math.sqrt(update_sq),
        }


class FlashSophiaH(SophiaH):
    """SophiaH variant optimized for lower Hessian-refresh overhead.

    The optimizer state matches `SophiaH`, but Hutchinson estimates can be
    streamed directly into the Hessian EMA instead of materializing a second
    full-model estimate list.
    """

    @torch.no_grad()
    def begin_hessian_update(self):
        for group in self.param_groups:
            beta2 = group["betas"][1]
            for p in group["params"]:
                if p.requires_grad:
                    self._init_state(p)["hessian"].mul_(beta2)

    @torch.no_grad()
    def accumulate_hessian_sample(
        self,
        hvp,
        probes,
        *,
        alpha: float,
        average_distributed: bool = False,
        bucket_bytes: int = 64 * 1024 * 1024,
    ):
        records = []
        flat_idx = 0
        for group in self.param_groups:
            beta2 = group["betas"][1]
            scale = (1.0 - beta2) * float(alpha)
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self._init_state(p)
                records.append((state["hessian"], hvp[flat_idx], probes[flat_idx], scale))
                flat_idx += 1

        if flat_idx != len(hvp) or flat_idx != len(probes):
            raise ValueError("mismatched number of Hessian-vector products or probes")

        if average_distributed:
            self._average_and_add_records_(records, bucket_bytes=bucket_bytes)
            return

        for hessian, hv, probe, scale in records:
            hessian.addcmul_(hv, probe, value=scale)

    @torch.no_grad()
    def _average_and_add_records_(self, records, *, bucket_bytes: int):
        if not (dist.is_available() and dist.is_initialized()):
            for hessian, hv, probe, scale in records:
                hessian.addcmul_(hv, probe, value=scale)
            return

        world_size = dist.get_world_size()
        bucket_elems = max(1, int(bucket_bytes) // 4)
        pending = []
        pending_elems = 0

        def flush():
            nonlocal pending, pending_elems
            if not pending:
                return
            flat = torch.cat([entry[3] for entry in pending])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
            offset = 0
            for hessian_flat, start, end, _values, scale in pending:
                n = end - start
                hessian_flat[start:end].add_(flat[offset : offset + n], alpha=scale)
                offset += n
            pending = []
            pending_elems = 0

        for hessian, hv, probe, scale in records:
            hessian_flat = hessian.view(-1)
            values = (hv * probe).float().reshape(-1)
            numel = values.numel()
            start = 0
            while start < numel:
                take = min(bucket_elems, numel - start)
                if pending and pending_elems + take > bucket_elems:
                    flush()
                end = start + take
                pending.append((hessian_flat, start, end, values[start:end], scale))
                pending_elems += take
                start = end
        flush()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, _ = group["betas"]
            lr = group["lr"]
            gamma = group["gamma"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self._init_state(p)
                grad = p.grad.float()
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                state["step"] += 1
                step = state["step"]

                p.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                denom = torch.empty_like(exp_avg)
                update = torch.empty_like(exp_avg)
                denom.copy_(hessian).clamp_(min=0.0).mul_(gamma).add_(eps)
                torch.abs(exp_avg, out=update)
                update.div_(denom).clamp_(max=1.0)
                torch.sign(exp_avg, out=denom)
                update.mul_(denom)

                step_size = lr / (1.0 - beta1**step)
                if update.dtype == p.dtype:
                    p.add_(update, alpha=-step_size)
                else:
                    p.add_(update.to(dtype=p.dtype), alpha=-step_size)


class AdaHessian(torch.optim.Optimizer):
    """AdaHessian update rule adapted to the trainer's explicit Hutchinson pass.

    The parameter update matches the public `davda54/ada-hessian` implementation,
    but the Hessian-trace estimates are supplied externally via `update_hessian()`
    so this harness can reuse the same microbatch/monolithic Hutchinson path that
    already exists for SophiaH.
    """

    def __init__(
        self,
        params,
        lr: float = 6e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        hessian_power: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid beta2: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight decay: {weight_decay}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"invalid hessian_power: {hessian_power}")

        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                hessian_power=hessian_power,
            ),
        )

    def _init_state(self, p: torch.Tensor):
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32)
            state["exp_hessian_diag_sq"] = torch.zeros_like(
                p,
                memory_format=torch.preserve_format,
                dtype=torch.float32,
            )
            state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32)
        return state

    @torch.no_grad()
    def update_hessian(self, hessian_estimates):
        flat_idx = 0
        for group in self.param_groups:
            beta2 = group["betas"][1]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self._init_state(p)
                h_est = hessian_estimates[flat_idx].float()
                flat_idx += 1
                state["hessian"].copy_(h_est)
                state["exp_hessian_diag_sq"].mul_(beta2).addcmul_(h_est, h_est, value=1.0 - beta2)

        if flat_idx != len(hessian_estimates):
            raise ValueError("mismatched number of hessian estimates")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            hessian_power = group["hessian_power"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self._init_state(p)
                grad = p.grad.float()
                exp_avg = state["exp_avg"]
                exp_hessian_diag_sq = state["exp_hessian_diag_sq"]

                state["step"] += 1
                step = state["step"]

                p.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = (exp_hessian_diag_sq / bias_correction2).pow(hessian_power / 2.0).add_(eps)
                step_size = lr / bias_correction1
                update = exp_avg / denom
                p.add_(update.to(dtype=p.dtype), alpha=-step_size)

    @torch.no_grad()
    def logging_stats(self) -> dict:
        momentum_sq = 0.0
        hessian_sq = 0.0
        hessian_l1 = 0.0
        hessian_ema_sq = 0.0
        update_sq = 0.0

        for group in self.param_groups:
            _, beta2 = group["betas"]
            eps = group["eps"]
            hessian_power = group["hessian_power"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self._init_state(p)
                step = max(1, state["step"])
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]
                exp_hessian_diag_sq = state["exp_hessian_diag_sq"]
                denom = (exp_hessian_diag_sq / (1.0 - beta2**step)).pow(hessian_power / 2.0).add_(eps)
                momentum_sq += exp_avg.pow(2).sum().item()
                hessian_sq += hessian.pow(2).sum().item()
                hessian_l1 += hessian.abs().sum().item()
                hessian_ema_sq += exp_hessian_diag_sq.pow(2).sum().item()
                update_sq += (exp_avg / denom).pow(2).sum().item()

        return {
            "optimizer/momentum_l2": math.sqrt(momentum_sq),
            "optimizer/hessian_l2": math.sqrt(hessian_sq),
            "optimizer/hessian_l1": hessian_l1,
            "optimizer/hessian_ema_l2": math.sqrt(hessian_ema_sq),
            "optimizer/precond_update_l2": math.sqrt(update_sq),
        }
