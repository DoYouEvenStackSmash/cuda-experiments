#ifndef COMMON_KERNELS_H
#define COMMON_KERNELS_H
#include <cuda_runtime.h>
extern "C" void conv_wrap(float* A,float *B, float *H, int hlen, int ylen,int width, int height,int lb, int span);
// extern "C" void conv(float* A,float *B, float *H, int hlen, int ylen,int width, int height, int lb, int span);
#endif // CUDA_KERNEL_H
