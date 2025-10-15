/*
 * Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
 * Distributed under the MIT License.
 *(See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)
 * Raul P. Pelaez 2023. Torch extensions to the torchmdnet library.
 * You can expose functions to python here which will be compatible with TorchScript.
 * Add your exports to the TORCH_LIBRARY macro below, see __init__.py to see how to access them from python.
 * The WITH_CUDA macro will be defined when compiling with CUDA support.
 */


#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#if defined(WITH_CUDA)
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif


extern "C" {
  /* Creates a dummy empty torchmdnet_extensions module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit_torchmdnet_extensions(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "torchmdnet_extensions",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

/* @brief Returns true if the current torch CUDA stream is capturing.
 * This function is required because the one available in torch is not compatible with TorchScript.
 * @return True if the current torch CUDA stream is capturing.
 */
bool is_current_stream_capturing() {
#if defined(WITH_CUDA)
  auto current_stream = at::cuda::getCurrentCUDAStream().stream();
  cudaStreamCaptureStatus capture_status;
  cudaError_t err = cudaStreamGetCaptureInfo(current_stream, &capture_status, nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
  return capture_status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive;
#else
  return false;
#endif
}

TORCH_LIBRARY(torchmdnet_extensions, m) {
    m.def("is_current_stream_capturing", is_current_stream_capturing);
    m.def("get_neighbor_pairs(str strategy, Tensor positions, Tensor batch, Tensor box_vectors, "
          "bool use_periodic, Scalar cutoff_lower, Scalar cutoff_upper, Scalar max_num_pairs, bool "
          "loop, bool include_transpose) -> (Tensor neighbors, Tensor distances, Tensor "
          "distance_vecs, Tensor num_pairs)");
    //The individual fwd and bkwd functions must be exposed in order to register their meta implementations python side.
    m.def("get_neighbor_pairs_fwd(str strategy, Tensor positions, Tensor batch, Tensor box_vectors, "
          "bool use_periodic, Scalar cutoff_lower, Scalar cutoff_upper, Scalar max_num_pairs, bool "
          "loop, bool include_transpose) -> (Tensor neighbors, Tensor distances, Tensor "
          "distance_vecs, Tensor num_pairs)");
    m.def("get_neighbor_pairs_bkwd(Tensor grad_edge_vec, Tensor grad_edge_weight, Tensor edge_index, "
	  "Tensor edge_vec, Tensor edge_weight, int num_atoms) -> Tensor");
}
