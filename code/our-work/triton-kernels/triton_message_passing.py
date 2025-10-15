import torch
import triton
import triton.language as tl


@triton.jit
def fused_index_select_index_add_kernel(
    x_ptr,
    edge_index_src_ptr,
    edge_index_dst_ptr,
    edge_weight_ptr,
    out_ptr,
    N, E, D,
    BLOCK_SIZE: tl.constexpr,
):
    edge_idx = tl.program_id(0)

    if edge_idx >= E:
        return

    src_idx = tl.load(edge_index_src_ptr + edge_idx)
    dst_idx = tl.load(edge_index_dst_ptr + edge_idx)
    weight = tl.load(edge_weight_ptr + edge_idx)

    for d_start in range(0, D, BLOCK_SIZE):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE)
        d_mask = d_offsets < D

        src_feats = tl.load(x_ptr + src_idx * D + d_offsets, mask=d_mask, other=0.0)
        msg = weight * src_feats
        tl.atomic_add(out_ptr + dst_idx * D + d_offsets, msg, mask=d_mask)


def message_passing_triton(x, edge_index, edge_weight, num_nodes):
    N, D = x.shape
    E = edge_index.shape[1]

    out = torch.zeros(num_nodes, D, dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128
    grid = (E,)

    fused_index_select_index_add_kernel[grid](
        x, edge_index[1], edge_index[0], edge_weight,
        out, num_nodes, E, D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


@triton.jit
def fused_tensor_message_passing_kernel(
    tensor_ptr,
    edge_index_src_ptr,
    edge_index_dst_ptr,
    factor_ptr,
    out_ptr,
    N, E, C, N_out,
    BLOCK_SIZE_C: tl.constexpr,
):
    edge_idx = tl.program_id(0)

    if edge_idx >= E:
        return

    src_idx = tl.load(edge_index_src_ptr + edge_idx)
    dst_idx = tl.load(edge_index_dst_ptr + edge_idx)

    for c_start in range(0, C, BLOCK_SIZE_C):
        c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
        c_mask = c_offsets < C

        for i in range(3):
            for j in range(3):
                tensor_offset = src_idx * C * 9 + c_offsets * 9 + i * 3 + j
                factor_offset = edge_idx * C + c_offsets

                tensor_val = tl.load(tensor_ptr + tensor_offset, mask=c_mask, other=0.0)
                factor_val = tl.load(factor_ptr + factor_offset, mask=c_mask, other=0.0)

                msg = factor_val * tensor_val

                out_offset = dst_idx * C * 9 + c_offsets * 9 + i * 3 + j
                tl.atomic_add(out_ptr + out_offset, msg, mask=c_mask)


def tensor_message_passing_triton(tensor, edge_index, factor, num_nodes):
    N, C, _, _ = tensor.shape
    E = edge_index.shape[1]

    tensor_flat = tensor.reshape(N, C * 9)
    factor_flat = factor.reshape(E, C)

    out_flat = torch.zeros(num_nodes, C * 9, dtype=tensor.dtype, device=tensor.device)

    BLOCK_SIZE_C = 32
    grid = (E,)

    fused_tensor_message_passing_kernel[grid](
        tensor_flat, edge_index[1], edge_index[0], factor_flat,
        out_flat, N, E, C, num_nodes,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )

    return out_flat.reshape(num_nodes, C, 3, 3)


@triton.jit
def fused_index_add_4d_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    E, N, C,
    BLOCK_SIZE_C: tl.constexpr,
):
    edge_idx = tl.program_id(0)

    if edge_idx >= E:
        return

    dst_idx = tl.load(index_ptr + edge_idx)

    for c_start in range(0, C, BLOCK_SIZE_C):
        c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
        c_mask = c_offsets < C

        for i in range(3):
            for j in range(3):
                src_offset = edge_idx * C * 9 + c_offsets * 9 + i * 3 + j
                dst_offset = dst_idx * C * 9 + c_offsets * 9 + i * 3 + j

                val = tl.load(src_ptr + src_offset, mask=c_mask, other=0.0)
                tl.atomic_add(out_ptr + dst_offset, val, mask=c_mask)


def index_add_4d_triton(src, index, num_nodes):
    E, C, _, _ = src.shape

    src_flat = src.reshape(E, C * 9)
    out_flat = torch.zeros(num_nodes, C * 9, dtype=src.dtype, device=src.device)

    BLOCK_SIZE_C = 32
    grid = (E,)

    fused_index_add_4d_kernel[grid](
        src_flat, index, out_flat,
        E, num_nodes, C,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )

    return out_flat.reshape(num_nodes, C, 3, 3)


@triton.jit
def fused_cutoff_message_kernel(
    x_ptr,
    edge_index_src_ptr,
    edge_index_dst_ptr,
    edge_weight_ptr,
    out_ptr,
    N, E, D,
    cutoff_lower, cutoff_upper,
    BLOCK_SIZE: tl.constexpr,
):
    edge_idx = tl.program_id(0)

    if edge_idx >= E:
        return

    src_idx = tl.load(edge_index_src_ptr + edge_idx)
    dst_idx = tl.load(edge_index_dst_ptr + edge_idx)
    dist = tl.load(edge_weight_ptr + edge_idx)

    cutoff_width = cutoff_upper - cutoff_lower
    if dist < cutoff_lower:
        cutoff = 1.0
    elif dist > cutoff_upper:
        cutoff = 0.0
    else:
        x_cutoff = (dist - cutoff_lower) / cutoff_width
        cutoff = 0.5 * (tl.cos(3.14159265359 * x_cutoff) + 1.0)

    if cutoff == 0.0:
        return

    for d_start in range(0, D, BLOCK_SIZE):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE)
        d_mask = d_offsets < D

        src_feats = tl.load(x_ptr + src_idx * D + d_offsets, mask=d_mask, other=0.0)
        msg = cutoff * src_feats
        tl.atomic_add(out_ptr + dst_idx * D + d_offsets, msg, mask=d_mask)


def cutoff_message_passing_triton(x, edge_index, edge_weight, num_nodes, cutoff_lower, cutoff_upper):
    N, D = x.shape
    E = edge_index.shape[1]

    out = torch.zeros(num_nodes, D, dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128
    grid = (E,)

    fused_cutoff_message_kernel[grid](
        x, edge_index[1], edge_index[0], edge_weight,
        out, num_nodes, E, D,
        cutoff_lower, cutoff_upper,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


@triton.jit
def fused_rbf_message_kernel(
    x_ptr,
    edge_index_src_ptr,
    edge_index_dst_ptr,
    edge_attr_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N, E, D, num_rbf,
    BLOCK_SIZE_D: tl.constexpr,
):
    edge_idx = tl.program_id(0)

    if edge_idx >= E:
        return

    src_idx = tl.load(edge_index_src_ptr + edge_idx)
    dst_idx = tl.load(edge_index_dst_ptr + edge_idx)

    for d_start in range(0, D, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D

        edge_proj = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
        for r in range(num_rbf):
            edge_val = tl.load(edge_attr_ptr + edge_idx * num_rbf + r)
            weight_vals = tl.load(weight_ptr + r * D + d_offsets, mask=d_mask, other=0.0)
            edge_proj += edge_val * weight_vals

        bias_vals = tl.load(bias_ptr + d_offsets, mask=d_mask, other=0.0)
        edge_proj += bias_vals

        src_feats = tl.load(x_ptr + src_idx * D + d_offsets, mask=d_mask, other=0.0)
        msg = edge_proj * src_feats

        tl.atomic_add(out_ptr + dst_idx * D + d_offsets, msg, mask=d_mask)


def rbf_message_passing_triton(x, edge_index, edge_attr, linear_weight, linear_bias, num_nodes):
    N, D = x.shape
    E, num_rbf = edge_attr.shape

    out = torch.zeros(num_nodes, D, dtype=x.dtype, device=x.device)

    BLOCK_SIZE_D = 64
    grid = (E,)

    fused_rbf_message_kernel[grid](
        x, edge_index[1], edge_index[0],
        edge_attr, linear_weight, linear_bias,
        out, num_nodes, E, D, num_rbf,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return out
