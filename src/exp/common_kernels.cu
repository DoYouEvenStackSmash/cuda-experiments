#include "cuda_runtime.h"
// #include "cuda_kernel.h"
#include "common_kernels.h"

__global__ void conv(float* A,float *B, float *H, int hlen, int ylen,int width, int height, int lb, int span) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // int row = blockIdx.y * blockDim.y + threadIdx.y;
  int i = col;
  if (col < ylen && col >= lb * width) {
    float sum = 0.0f;
    for (int j = 0; j < hlen; j++) {
      if (i-j*width < 0)
        break;
      sum = sum + (float) H[j] * A[i - j * width];
    }
    B[i] = sum;
  }
}

__global__ void obs_mask(float *A, float *B, int ylen, int width, int height, int lb, int span) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < ylen) {
    B[col] = A[col] < 0.5 ? 1 : 0;
  }
}

// __global__ void const_conv(float* A,float *B, int ylen,int width, int height) {
//   int col = blockIdx.x * blockDim.x + threadIdx.x;
//   // int row = blockIdx.y * blockDim.y + threadIdx.y;
//   int i = col;
//   if (col < ylen) {
//     float sum = 0.0f;
//     for (int j = 0; j < KERNEL_LENGTH; j++) {
//       if (i-j*width < 0)break;
//       sum = sum + (float) c_Kernel[j] * A[i - j * width];
//     }
//     B[col] += sum;
//   }
// }

extern "C" void conv_wrap(float* A,float *B, float *H, int hlen, int ylen,int width, int height,int lb,int span) {
  int threads = min(1024,(height + height%32));
  int off_t = 0;
  if (threads == 1024) {
    off_t = height - threads;
  }

  // int blocks = ceil(ylen / min(height, threads))
  conv<<<(width+off_t+width%32),min(1024,(height + height%32))>>>(A,B,H, hlen, ylen,width,height,lb,span);
}

extern "C" void obs_mask_wrap(float *A, float *B, int ylen, int width, int height, int lb, int span) {
  int threads = min(1024,(height + height%32));
  int off_t = 0;
  if (threads == 1024) {
    off_t = height - threads;
  }
  obs_mask<<<(width+off_t+width%32),min(1024,(height + height%32))>>>(A,  B,  ylen,  width,  height,  lb,  span); 
}
// extern "C" void const_conv_wrap(float* A,float *B, int ylen,int width, int height) {
//   const_conv<<<(width+width%32),min(1024,(height + (32 - height%32)))>>>(A,B, ylen,width,height);
// }