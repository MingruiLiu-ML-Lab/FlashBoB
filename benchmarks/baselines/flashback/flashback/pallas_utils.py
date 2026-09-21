import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import enum

class Precision(enum.Enum):
    TF32_ROUND = 0 # Go to TF32 by *rounding* FP32 values
    TF32_TRUNC = 1 # Go to TF32 by *truncating* FP32 values
    FP32 = 2
    FP16 = 3
    BF16 = 4

def display_error(x, x_ref, name):
    abs_error = jnp.abs(x - x_ref).mean()
    rel_error = abs_error / jnp.abs(x_ref).mean()

    error_message = f"{name}: abs = {abs_error:.6e}, rel = {rel_error:.6e}"

    if rel_error > 0.001:
        print(f"\033[91mERROR: {error_message}\033[0m")
    else:
        print(f"\033[92mOK: {error_message}\033[0m")

def round_tf32(x):
    ASM = "cvt.rna.tf32.f32 $0, $1;"
    [result] = plgpu.elementwise_inline_asm(
            ASM,
            args=[x],
            constraints="=r, r",
            pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, x.dtype)]
    )
    return result

def tile_dot(a, b, precision):
    if precision == Precision.TF32_ROUND:
        return pl.dot(round_tf32(a), round_tf32(b), precision="tensorfloat32")
    elif precision == Precision.TF32_TRUNC:
        return pl.dot(a.astype(jnp.float32), b.astype(jnp.float32), precision="tensorfloat32")
    elif precision == Precision.FP32:
        return pl.dot(a.astype(jnp.float32), b.astype(jnp.float32), precision="highest")
    elif precision == Precision.BF16:
        return pl.dot(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16))
    elif precision == Precision.FP16:
        return pl.dot(a.astype(jnp.float16), b.astype(jnp.float16))
    else:
        raise ValueError(f"Invalid precision: {precision}")

def triton_compiler_params(*, num_warps, num_stages):
    return plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages)

def accumulate(ref, slices, val, atomic_add):
    if atomic_add:
        pl.atomic_add(ref, slices, val.astype(ref.dtype))
    else:
        old = pl.load(ref, slices)
        pl.store(ref, slices, (old + val).astype(ref.dtype))
