#include "pybind_kernel.h"
__global__ void c_multiply(double* buf1, double* buf2,int bufcount, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < buflen) {
    buf1[col] = buf1[col] * buf2[col];
  }
}
void cuda_multiply(double* ptr3, double* ptr1, double* ptr2, int bufcount, int buflen) {
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