import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torchmdnet.triton_kernels import (
    decompose_tensor_triton,
    tensor_norm_triton,
    vector_to_symtensor_triton,
    vector_to_skewtensor_triton,
)


def simulate_tensornet_forward_pass(n_atoms, n_neighbors_per_atom):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        return

    positions = torch.randn(n_atoms, 3, device=device)
    n_edges = n_atoms * n_neighbors_per_atom
    edge_vectors = torch.randn(n_edges, 3, device=device)
    edge_vectors = edge_vectors / (edge_vectors.norm(dim=1, keepdim=True) + 1e-8)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    symmetric_tensors = vector_to_symtensor_triton(edge_vectors)
    skew_tensors = vector_to_skewtensor_triton(edge_vectors)
    end.record()
    torch.cuda.synchronize()
    tensor_construction_time = start.elapsed_time(end)

    combined_tensors = symmetric_tensors + 0.5 * skew_tensors

    start.record()
    I_comp, A_comp, S_comp = decompose_tensor_triton(combined_tensors)
    end.record()
    torch.cuda.synchronize()
    decomposition_time = start.elapsed_time(end)

    start.record()
    sym_norms = tensor_norm_triton(S_comp)
    antisym_norms = tensor_norm_triton(A_comp)
    end.record()
    torch.cuda.synchronize()
    norm_time = start.elapsed_time(end)

    total_time = tensor_construction_time + decomposition_time + norm_time
    operations_per_atom = 5
    throughput = (n_atoms * n_neighbors_per_atom * operations_per_atom) / (total_time / 1000)

    return {
        "total_time_ms": total_time,
        "tensor_construction_ms": tensor_construction_time,
        "decomposition_ms": decomposition_time,
        "norm_ms": norm_time,
        "throughput_ops_per_sec": throughput,
        "symmetric_tensors": symmetric_tensors,
        "I_comp": I_comp,
        "A_comp": A_comp,
        "S_comp": S_comp,
        "sym_norms": sym_norms,
        "antisym_norms": antisym_norms,
    }


def demonstrate_physical_properties():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        return

    N = 1000
    test_vectors = torch.randn(N, 3, device=device)
    test_tensors = torch.randn(N, 3, 3, device=device)

    S = vector_to_symtensor_triton(test_vectors)
    is_symmetric = torch.allclose(S, S.transpose(-2, -1), rtol=1e-5)

    traces = S.diagonal(dim1=-2, dim2=-1).sum(-1)
    is_traceless = torch.allclose(traces, torch.zeros_like(traces), atol=1e-5)

    A = vector_to_skewtensor_triton(test_vectors)
    is_antisymmetric = torch.allclose(A, -A.transpose(-2, -1), rtol=1e-5)

    diagonals = A.diagonal(dim1=-2, dim2=-1)
    is_zero_diag = torch.allclose(diagonals, torch.zeros_like(diagonals), atol=1e-5)

    I, A, S = decompose_tensor_triton(test_tensors)
    reconstructed = I + A + S
    is_complete = torch.allclose(reconstructed, test_tensors, rtol=1e-4)

    total_norm = tensor_norm_triton(test_tensors)
    I_norm = tensor_norm_triton(I)
    A_norm = tensor_norm_triton(A)
    S_norm = tensor_norm_triton(S)
    component_norm_sum = I_norm + A_norm + S_norm
    norms_match = torch.allclose(total_norm, component_norm_sum, rtol=1e-3)


def main():
    demonstrate_physical_properties()

    simulate_tensornet_forward_pass(n_atoms=50, n_neighbors_per_atom=32)
    simulate_tensornet_forward_pass(n_atoms=500, n_neighbors_per_atom=32)
    simulate_tensornet_forward_pass(n_atoms=5000, n_neighbors_per_atom=32)


if __name__ == "__main__":
    main()
