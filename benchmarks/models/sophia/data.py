from collections import deque

import numpy as np
import tiktoken
import torch

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from benchmarks.models.sophia.runtime import normalize_optional_str


DATASET_PRESETS = {
    "fineweb": {
        "dataset_id": "VisionTheta/fineweb-1B",
        "dataset_name": None,
    },
    "openwebtext": {
        "dataset_id": "openwebtext",
        "dataset_name": None,
    },
}


def encode_text_for_training(enc, text: str, *, backend: str = "tiktoken") -> np.ndarray:
    if backend == "tiktoken":
        tokens = enc.encode(text, allowed_special=enc.special_tokens_set)
    else:
        tokens = enc.encode(text, add_special_tokens=False)
    return np.asarray(tokens, dtype=np.int32)


class TokenChunkQueue:
    """Compact token queue backed by numpy chunks instead of Python ints."""

    def __init__(self):
        self._chunks = deque()
        self._head_offset = 0
        self.size = 0

    def append(self, chunk: np.ndarray):
        if chunk.size == 0:
            return
        self._chunks.append(chunk)
        self.size += int(chunk.size)

    def pop(self, count: int) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be >= 0")
        if count > self.size:
            raise ValueError("not enough buffered tokens")

        out = np.empty(count, dtype=np.int32)
        written = 0

        while written < count:
            chunk = self._chunks[0]
            start = self._head_offset
            available = int(chunk.size) - start
            take = min(count - written, available)
            out[written : written + take] = chunk[start : start + take]
            written += take
            self.size -= take

            if take == available:
                self._chunks.popleft()
                self._head_offset = 0
            else:
                self._head_offset += take

        return out


def resolve_dataset_args(args):
    if args.dataset != "custom":
        preset = DATASET_PRESETS[args.dataset]
        if normalize_optional_str(args.dataset_id) is None:
            args.dataset_id = preset["dataset_id"]
        if normalize_optional_str(args.dataset_name) is None and preset["dataset_name"] is not None:
            args.dataset_name = preset["dataset_name"]

    dataset_id = normalize_optional_str(args.dataset_id)
    if dataset_id is None:
        raise ValueError("dataset_id must be provided when --dataset=custom")

    args.dataset_id = dataset_id
    args.dataset_name = normalize_optional_str(args.dataset_name) or ""
    args.dataset_revision = normalize_optional_str(args.dataset_revision) or ""
    args.dataset_data_dir = normalize_optional_str(args.dataset_data_dir) or ""
    return args


class PackedTextIterableDataset(IterableDataset):
    def __init__(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        dataset_revision: str,
        dataset_data_dir: str,
        tokenizer_name: str,
        tokenizer_backend: str,
        tokenizer_revision: str,
        split_kind: str,
        val_buckets: int,
        seed: int,
        shuffle_buffer: int,
        batch_size: int,
        block_size: int,
        rank: int,
        world_size: int,
        streaming: bool,
        hf_hub_cache: str,
    ):
        if split_kind not in {"train", "val"}:
            raise ValueError("split_kind must be 'train' or 'val'")

        self.dataset_id = dataset_id
        self.dataset_name = normalize_optional_str(dataset_name)
        self.dataset_revision = normalize_optional_str(dataset_revision)
        self.dataset_data_dir = normalize_optional_str(dataset_data_dir)
        self.tokenizer_name = tokenizer_name
        self.tokenizer_backend = tokenizer_backend
        self.tokenizer_revision = normalize_optional_str(tokenizer_revision)
        self.split_kind = split_kind
        self.val_buckets = val_buckets
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.streaming = streaming
        self.hf_hub_cache = normalize_optional_str(hf_hub_cache)

    @staticmethod
    def _stable_bucket(text: str, num_buckets: int = 10000) -> int:
        import hashlib

        digest = hashlib.sha1(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % num_buckets

    def _keep(self, example) -> bool:
        key = example.get("id")
        if key is None:
            key = example.get("text", "")
        is_val = self._stable_bucket(str(key)) < self.val_buckets
        return is_val if self.split_kind == "val" else not is_val

    def _load_kwargs(self) -> dict:
        kwargs = {
            "path": self.dataset_id,
            "split": "train",
            "streaming": self.streaming,
        }
        if self.dataset_name is not None:
            kwargs["name"] = self.dataset_name
        if self.dataset_revision is not None:
            kwargs["revision"] = self.dataset_revision
        if self.dataset_data_dir is not None:
            kwargs["data_dir"] = self.dataset_data_dir
        if self.hf_hub_cache is not None:
            kwargs["cache_dir"] = self.hf_hub_cache
        return kwargs

    def _build_dataset(self, worker_id: int, num_workers: int, epoch: int):
        ds = load_dataset(**self._load_kwargs())

        if self.split_kind == "train" and self.shuffle_buffer > 0:
            ds = ds.shuffle(seed=self.seed + epoch, buffer_size=self.shuffle_buffer)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

        if self.world_size > 1:
            ds = split_dataset_by_node(ds, rank=self.rank, world_size=self.world_size)

        if num_workers > 1:
            ds = split_dataset_by_node(ds, rank=worker_id, world_size=num_workers)

        return ds

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers

        if self.tokenizer_backend == "tiktoken":
            enc = tiktoken.get_encoding(self.tokenizer_name)
            eot = enc.eot_token
        else:
            from transformers import AutoTokenizer

            enc = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                revision=self.tokenizer_revision,
                cache_dir=self.hf_hub_cache,
            )
            eot = enc.eos_token_id
            if eot is None:
                raise ValueError("the Hugging Face tokenizer must define eos_token_id")
        need = self.batch_size * (self.block_size + 1)
        eot_chunk = np.asarray([eot], dtype=np.int32)
        buffer = TokenChunkQueue()
        epoch = 0

        while True:
            ds = self._build_dataset(worker_id=worker_id, num_workers=num_workers, epoch=epoch)

            for example in ds:
                if not self._keep(example):
                    continue

                text = example.get("text", "")
                if not text:
                    continue

                tokens = encode_text_for_training(
                    enc, text, backend=self.tokenizer_backend
                )
                if tokens.size == 0:
                    continue

                buffer.append(tokens.astype(np.int32, copy=False))
                buffer.append(eot_chunk)

                while buffer.size >= need:
                    chunk = buffer.pop(need)
                    batch = torch.tensor(chunk, dtype=torch.long).view(self.batch_size, self.block_size + 1)
                    yield batch[:, :-1], batch[:, 1:]

            epoch += 1


class DeviceBatchStream:
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.iterator = iter(loader)

    def next_batch(self):
        idx, targets = next(self.iterator)
        idx = idx.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        return idx, targets


def create_batch_loader(
    *,
    args,
    cfg,
    split_kind: str,
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int | None = None,
):
    dataset = PackedTextIterableDataset(
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        dataset_revision=args.dataset_revision,
        dataset_data_dir=args.dataset_data_dir,
        tokenizer_name=args.tokenizer_name,
        tokenizer_backend=args.tokenizer_backend,
        tokenizer_revision=args.tokenizer_revision,
        split_kind=split_kind,
        val_buckets=args.val_buckets,
        seed=args.seed if seed is None else seed,
        shuffle_buffer=args.shuffle_buffer,
        batch_size=batch_size,
        block_size=cfg.block_size,
        rank=rank,
        world_size=world_size,
        streaming=args.streaming,
        hf_hub_cache=args.hf_hub_cache,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "pin_memory": True,
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
    }

    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    return DataLoader(**loader_kwargs)
