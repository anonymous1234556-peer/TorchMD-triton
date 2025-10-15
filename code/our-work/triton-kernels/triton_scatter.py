import torch
import triton
import triton.language as tl


@triton.jit
def fused_scatter_add_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    N,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid // tl.cdiv(D, BLOCK_SIZE)
    col_block = pid % tl.cdiv(D, BLOCK_SIZE)

    if row_idx >= N:
        return

    idx = tl.load(index_ptr + row_idx)

    col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    src_offsets = row_idx * D + col_offsets
    out_offsets = idx * D + col_offsets

    src_vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

    tl.atomic_add(out_ptr + out_offsets, src_vals, mask=mask)


def scatter_add_triton(src, index, dim_size):
    N, D = src.shape
    out = torch.zeros(dim_size, D, dtype=src.dtype, device=src.device)

    BLOCK_SIZE = 128
    grid = ((N * triton.cdiv(D, BLOCK_SIZE)),)

    fused_scatter_add_kernel[grid](
        src, index, out, N, D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


@triton.jit
def fused_index_select_mul_kernel(
    x_ptr,
    index_ptr,
    weight_ptr,
    out_ptr,
    N,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid

    if row_idx >= N:
        return

    idx = tl.load(index_ptr + row_idx)
    w = tl.load(weight_ptr + row_idx)

    for d in range(0, D, BLOCK_SIZE):
        offsets = d + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D

        vals = tl.load(x_ptr + idx * D + offsets, mask=mask, other=0.0)

        result = vals * w

        tl.store(out_ptr + row_idx * D + offsets, result, mask=mask)


def index_select_mul_triton(x, index, weight):
    N = index.shape[0]
    D = x.shape[1]

    out = torch.empty(N, D, dtype=x.dtype, device=x.device)

    grid = (N,)

    fused_index_select_mul_kernel[grid](
        x, index, weight, out, N, D,
        BLOCK_SIZE=128,
    )

    return out
