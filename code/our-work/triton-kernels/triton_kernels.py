import torch
import triton
import triton.language as tl


@triton.jit
def fused_tensor_decomposition_kernel(
    tensor_ptr,
    I_ptr,
    A_ptr,
    S_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid >= N:
        return

    base_offset = pid * 9

    t00 = tl.load(tensor_ptr + base_offset + 0)
    t01 = tl.load(tensor_ptr + base_offset + 1)
    t02 = tl.load(tensor_ptr + base_offset + 2)
    t10 = tl.load(tensor_ptr + base_offset + 3)
    t11 = tl.load(tensor_ptr + base_offset + 4)
    t12 = tl.load(tensor_ptr + base_offset + 5)
    t20 = tl.load(tensor_ptr + base_offset + 6)
    t21 = tl.load(tensor_ptr + base_offset + 7)
    t22 = tl.load(tensor_ptr + base_offset + 8)

    trace = (t00 + t11 + t22) / 3.0

    i00 = trace
    i11 = trace
    i22 = trace
    i_off_diag = 0.0

    a00 = 0.0
    a01 = (t01 - t10) * 0.5
    a02 = (t02 - t20) * 0.5
    a10 = (t10 - t01) * 0.5
    a11 = 0.0
    a12 = (t12 - t21) * 0.5
    a20 = (t20 - t02) * 0.5
    a21 = (t21 - t12) * 0.5
    a22 = 0.0

    s00 = (t00 + t00) * 0.5 - trace
    s01 = (t01 + t10) * 0.5
    s02 = (t02 + t20) * 0.5
    s10 = (t10 + t01) * 0.5
    s11 = (t11 + t11) * 0.5 - trace
    s12 = (t12 + t21) * 0.5
    s20 = (t20 + t02) * 0.5
    s21 = (t21 + t12) * 0.5
    s22 = (t22 + t22) * 0.5 - trace

    tl.store(I_ptr + base_offset + 0, i00)
    tl.store(I_ptr + base_offset + 1, i_off_diag)
    tl.store(I_ptr + base_offset + 2, i_off_diag)
    tl.store(I_ptr + base_offset + 3, i_off_diag)
    tl.store(I_ptr + base_offset + 4, i11)
    tl.store(I_ptr + base_offset + 5, i_off_diag)
    tl.store(I_ptr + base_offset + 6, i_off_diag)
    tl.store(I_ptr + base_offset + 7, i_off_diag)
    tl.store(I_ptr + base_offset + 8, i22)

    tl.store(A_ptr + base_offset + 0, a00)
    tl.store(A_ptr + base_offset + 1, a01)
    tl.store(A_ptr + base_offset + 2, a02)
    tl.store(A_ptr + base_offset + 3, a10)
    tl.store(A_ptr + base_offset + 4, a11)
    tl.store(A_ptr + base_offset + 5, a12)
    tl.store(A_ptr + base_offset + 6, a20)
    tl.store(A_ptr + base_offset + 7, a21)
    tl.store(A_ptr + base_offset + 8, a22)

    tl.store(S_ptr + base_offset + 0, s00)
    tl.store(S_ptr + base_offset + 1, s01)
    tl.store(S_ptr + base_offset + 2, s02)
    tl.store(S_ptr + base_offset + 3, s10)
    tl.store(S_ptr + base_offset + 4, s11)
    tl.store(S_ptr + base_offset + 5, s12)
    tl.store(S_ptr + base_offset + 6, s20)
    tl.store(S_ptr + base_offset + 7, s21)
    tl.store(S_ptr + base_offset + 8, s22)


@triton.jit
def tensor_norm_kernel(
    tensor_ptr,
    norm_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    norm_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in range(9):
        tensor_offsets = offsets * 9 + i
        t_vals = tl.load(tensor_ptr + tensor_offsets, mask=mask, other=0.0)
        norm_sq += t_vals * t_vals

    tl.store(norm_ptr + offsets, norm_sq, mask=mask)


@triton.jit
def vector_to_symtensor_kernel(
    vector_ptr,
    tensor_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid >= N:
        return

    v0 = tl.load(vector_ptr + pid * 3 + 0)
    v1 = tl.load(vector_ptr + pid * 3 + 1)
    v2 = tl.load(vector_ptr + pid * 3 + 2)

    trace = (v0 * v0 + v1 * v1 + v2 * v2) / 3.0

    t00 = v0 * v0 - trace
    t01 = v0 * v1
    t02 = v0 * v2
    t10 = v1 * v0
    t11 = v1 * v1 - trace
    t12 = v1 * v2
    t20 = v2 * v0
    t21 = v2 * v1
    t22 = v2 * v2 - trace

    base_offset = pid * 9
    tl.store(tensor_ptr + base_offset + 0, t00)
    tl.store(tensor_ptr + base_offset + 1, t01)
    tl.store(tensor_ptr + base_offset + 2, t02)
    tl.store(tensor_ptr + base_offset + 3, t10)
    tl.store(tensor_ptr + base_offset + 4, t11)
    tl.store(tensor_ptr + base_offset + 5, t12)
    tl.store(tensor_ptr + base_offset + 6, t20)
    tl.store(tensor_ptr + base_offset + 7, t21)
    tl.store(tensor_ptr + base_offset + 8, t22)


@triton.jit
def vector_to_skewtensor_kernel(
    vector_ptr,
    tensor_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid >= N:
        return

    v0 = tl.load(vector_ptr + pid * 3 + 0)
    v1 = tl.load(vector_ptr + pid * 3 + 1)
    v2 = tl.load(vector_ptr + pid * 3 + 2)

    zero = 0.0

    t00 = zero
    t01 = -v2
    t02 = v1
    t10 = v2
    t11 = zero
    t12 = -v0
    t20 = -v1
    t21 = v0
    t22 = zero

    base_offset = pid * 9
    tl.store(tensor_ptr + base_offset + 0, t00)
    tl.store(tensor_ptr + base_offset + 1, t01)
    tl.store(tensor_ptr + base_offset + 2, t02)
    tl.store(tensor_ptr + base_offset + 3, t10)
    tl.store(tensor_ptr + base_offset + 4, t11)
    tl.store(tensor_ptr + base_offset + 5, t12)
    tl.store(tensor_ptr + base_offset + 6, t20)
    tl.store(tensor_ptr + base_offset + 7, t21)
    tl.store(tensor_ptr + base_offset + 8, t22)


def decompose_tensor_triton(tensor):
    assert tensor.is_cuda
    original_shape = tensor.shape

    if tensor.ndim == 3 and tensor.shape[1:] == (3, 3):
        N = tensor.shape[0]
        tensor_flat = tensor.reshape(N, 9).contiguous()
        reshape_output = lambda x: x.reshape(N, 3, 3)
    elif tensor.ndim == 4 and tensor.shape[2:] == (3, 3):
        batch, channels = tensor.shape[:2]
        N = batch * channels
        tensor_flat = tensor.reshape(N, 9).contiguous()
        reshape_output = lambda x: x.reshape(batch, channels, 3, 3)
    else:
        N = tensor.shape[0]
        tensor_flat = tensor.reshape(N, -1).contiguous()
        reshape_output = lambda x: x.reshape(*tensor.shape)

    I = torch.empty_like(tensor_flat)
    A = torch.empty_like(tensor_flat)
    S = torch.empty_like(tensor_flat)

    grid = (N,)

    fused_tensor_decomposition_kernel[grid](
        tensor_flat, I, A, S, N,
        BLOCK_SIZE=128,
    )

    return reshape_output(I), reshape_output(A), reshape_output(S)


def tensor_norm_triton(tensor):
    if tensor.ndim == 1:
        return tensor
    elif tensor.ndim == 2 and tensor.shape[1] == 9:
        N = tensor.shape[0]
        tensor_flat = tensor.contiguous()
    elif tensor.ndim == 2 and tensor.shape[1] == 3:
        return (tensor ** 2).sum(dim=1)
    elif tensor.ndim == 3 and tensor.shape[1:] == (3, 3):
        N = tensor.shape[0]
        tensor_flat = tensor.reshape(N, 9).contiguous()
    elif tensor.ndim == 4 and tensor.shape[2:] == (3, 3):
        batch_size, channels = tensor.shape[:2]
        tensor_reshaped = tensor.reshape(batch_size * channels, 9).contiguous()

        N = batch_size * channels
        norm = torch.empty(N, dtype=tensor.dtype, device=tensor.device)
        BLOCK_SIZE = 128
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        tensor_norm_kernel[grid](tensor_reshaped, norm, N, BLOCK_SIZE=BLOCK_SIZE)

        return norm.reshape(batch_size, channels)
    else:
        N = tensor.shape[0]
        tensor_flat = tensor.reshape(N, -1).contiguous()

    assert tensor.is_cuda

    norm = torch.empty(N, dtype=tensor.dtype, device=tensor.device)

    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    tensor_norm_kernel[grid](
        tensor_flat, norm, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return norm


def vector_to_symtensor_triton(vector):
    assert vector.ndim == 2 and vector.shape[1] == 3
    assert vector.is_cuda

    N = vector.shape[0]
    vector_flat = vector.contiguous()

    tensor = torch.empty(N, 9, dtype=vector.dtype, device=vector.device)

    grid = (N,)

    vector_to_symtensor_kernel[grid](
        vector_flat, tensor, N,
        BLOCK_SIZE=128,
    )

    return tensor.reshape(N, 3, 3)


def vector_to_skewtensor_triton(vector):
    assert vector.ndim == 2 and vector.shape[1] == 3
    assert vector.is_cuda

    N = vector.shape[0]
    vector_flat = vector.contiguous()

    tensor = torch.empty(N, 9, dtype=vector.dtype, device=vector.device)

    grid = (N,)

    vector_to_skewtensor_kernel[grid](
        vector_flat, tensor, N,
        BLOCK_SIZE=128,
    )

    return tensor.reshape(N, 3, 3)
