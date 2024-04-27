#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pybind_kernel.h"
// #include <stdio.h>
// #include <stdlib.h>
#include <cuda_runtime.h>
// // Utilities and system includes
// #include <helper_functions.h>
// #include <helper_cuda.h>
namespace py = pybind11;

__global__ void c_multiply(double* buf1, double* buf2,int bufcount, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < buflen) {
    buf1[col] = buf1[col] * buf2[col];
  }
}
void cuda_multiply(double* ptr3, double* ptr1, double* ptr2, int bufcount, int buflen) {
  devID = gpuGetMaxGflopsDeviceId();
  checkCudaErrors(cudaSetDevice(devID));
  int major = 0, minor = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
          devID, _ConvertSMVer2ArchName(major, minor), major, minor);
  double *buf1_gpu;
  double *buf2_gpu;
  cudaMalloc((void**)&buf1_gpu, buflen * sizeof(double));
  cudaMemcpy(buf1_gpu, ptr1, buflen * sizeof(double),cudaMemcpyHostToDevice);
  cudaMalloc((void**)&buf2_gpu, buflen * sizeof(double));
  cudaMemcpy(buf2_gpu, ptr2, buflen * sizeof(double),cudaMemcpyHostToDevice);
  c_multiply<<<bufcount, buflen>>>(buf1_gpu, buf2_gpu, bufcount, buflen);
  cudaMemcpy(ptr3, buf1_gpu, buflen * sizeof(double),cudaMemcpyDeviceToHost);
  cudaFree(buf1_gpu);
  cudaFree(buf2_gpu);
}
// __global__ void c_multiply(double* buf1, double* buf2,int bufcount, int buflen) {
//   int col = blockIdx.x * blockDim.x + threadIdx.x;
//   if (col < buflen) {
//     buf1[col] = buf1[col] * buf2[col];
//   }
// }
// void cuda_multiply(double* buf1, double* buf2, int bufcount, int buflen) {
//   c_multiply<<<bufcount, buflen>>> (buf1, buf2, bufcount, buflen);
// }
// Function to perform array multiplication
py::array_t<double> array_multiply(py::array_t<double> arr1, py::array_t<double> arr2) {
    // Ensure both arrays have the same shape
    if (arr1.shape(0) != arr2.shape(0) || arr1.shape(1) != arr2.shape(1)) {
        throw std::runtime_error("Input arrays must have the same shape");
    }

    // Get pointers to the data
    auto buf1 = arr1.request(), buf2 = arr2.request();
    double *ptr1 = (double *)buf1.ptr, *ptr2 = (double *)buf2.ptr;

    // Create a new NumPy array to store the result
    py::array_t<double> result({buf1.shape[0], buf1.shape[1]});
    auto buf3 = result.request();
    double *ptr3 = (double *)buf3.ptr;
    
    cuda_multiply(ptr3, ptr1, ptr2, 1, buf1.shape[0]);

    // Perform element-wise multiplication
    return result;
}

// Binding code
PYBIND11_MODULE(mult_example, m) {
    m.def("array_multiply", &array_multiply, "Multiply two numpy arrays");
}
