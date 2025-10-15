/*
 * Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
 * Distributed under the MIT License.
 * (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)
 * Raul P. Pelaez 2023
 */
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "neighbors_cuda_brute.cuh"
#include "neighbors_cuda_cell.cuh"
#include "neighbors_cuda_shared.cuh"

static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
forward_impl_cuda(const std::string& strategy, const at::Tensor& positions, const at::Tensor& batch,
                  const at::Tensor& in_box_vectors, bool use_periodic, const at::Scalar& cutoff_lower,
                  const at::Scalar& cutoff_upper, const at::Scalar& max_num_pairs, bool loop,
                  bool include_transpose) {
    auto kernel = forward_brute;
    if (positions.size(0) >= 32768 && strategy == "brute") {
        kernel = forward_shared;
    }
    if (strategy == "brute") {
    } else if (strategy == "cell") {
        kernel = forward_cell;
    } else if (strategy == "shared") {
        kernel = forward_shared;
    } else {
        throw std::runtime_error("Unknown kernel name");
    }
    return kernel(positions, batch, in_box_vectors, use_periodic, cutoff_lower, cutoff_upper,
                  max_num_pairs, loop, include_transpose);
}

// We only need to register the CUDA version of the forward function here. This way we can avoid
// compiling this file in CPU-only mode The rest of the registrations take place in
// neighbors_cpu.cpp
TORCH_LIBRARY_IMPL(torchmdnet_extensions, CUDA, m) {
    m.impl("get_neighbor_pairs_fwd", forward_impl_cuda);
}
