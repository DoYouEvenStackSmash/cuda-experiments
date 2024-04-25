#include <stdio.h>
#include <cuda_runtime.h>

#include "common_kernels.h"

// Utilities and system includes
// #include <helper_functions.h>
// #include <helper_cuda.h>

__global__ void conv(float* A,float *B, float *H, int hlen, int ylen,int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // int row = blockIdx.y * blockDim.y + threadIdx.y;
  int i = col;
  if (col < ylen) {
    float sum = 0.0f;
    for (int j = 0; j < hlen; j++) {
      if (i-j*width < 0)break;
      sum = sum + (float) H[j] * A[i - j * width];
    }
    B[col] += sum;
  }
}

extern "C" void conv_wrap(float* A,float *B, float *H, int hlen, int ylen,int width, int height) {
  conv<<<(w+w%32),min(1024,(h + h%32)),0,stream>>>(A_gpu,B_gpu,H_gpu, flen, ylen,w,h);
}
// int main(int argc, char** argv) {
//   StopWatchInterface *hTimer = NULL;

//   // Use command-line specified CUDA device, otherwise use device with highest
//   // Gflops/s
//   findCudaDevice(argc, (const char **)argv);

//   sdkCreateTimer(&hTimer);

//   float *A, *B, *H;
//   float *A_gpu, *B_gpu,*H_gpu;
//   float *B_cpu;

//   int w = 10000;
//   int h = 10000; 
//   int ylen = w * h;
//   int flen = 30;
  
//   A = (float *)malloc(w * h * sizeof(float));
//   B = (float *)malloc(w * h * sizeof(float));
//   B_cpu = (float *)malloc(w * h * sizeof(float));
//   H = (float *)malloc(flen * sizeof(float));
  
//   for (int i = 0; i < flen; i++)
//     H[i] = 1.0f;
  
//   for (int i = 0; i < w; i++) {
//     for (int j = 0; j < h; j++) {
//       A[i * h +j] = (float) i * h + j;
//       B[i * h + j] = 0.0f;
//       B_cpu[i * h + j] = 0.0f;
//     }
//   }

//   cudaMalloc((void **) &A_gpu, w * h * sizeof(float));
//   cudaMalloc((void **) &B_gpu, w * h * sizeof(float));
//   cudaMalloc((void **) &H_gpu, flen * sizeof(float));

//   cudaMemcpy(A_gpu, A, w * h * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(B_gpu, B, w * h* sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(H_gpu, H, flen * sizeof(float), cudaMemcpyHostToDevice);
//   cudaStream_t stream;

//   cudaEvent_t start, stop;
//   checkCudaErrors(cudaEventCreate(&start));
//   checkCudaErrors(cudaEventCreate(&stop));

//   checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//   checkCudaErrors(cudaStreamSynchronize(stream));
//   checkCudaErrors(cudaEventRecord(start, stream));
  
//   conv<<<(w+w%32),min(1024,(h + h%32)),0,stream>>>(A_gpu,B_gpu,H_gpu, flen, ylen,w,h);
//   // Record the stop event
//   checkCudaErrors(cudaEventRecord(stop, stream));

//   // Wait for the stop event to complete
//   checkCudaErrors(cudaEventSynchronize(stop));
//   float msecTotal = 0.0f;
//   checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
//   // Compute and print the performance
//   int iter = 1;
//   float msecPerMatrixMul = msecTotal/iter;
//   double flopsPerMatrixMul = 2.0 * static_cast<double>(w) *
//                              static_cast<double>(h) * static_cast<double>(flen) * iter;
//   double gigaFlops =
//       (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
//   printf(
//       "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
//       " WorkgroupSize= %u threads/block\n",
//       gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, 32);
//   checkCudaErrors(cudaDeviceSynchronize());
//   sdkResetTimer(&hTimer);
//   sdkStartTimer(&hTimer);
//   cpu_convolve(A,B_cpu,H, flen, ylen,w,h);
//   sdkStopTimer(&hTimer);
//   sdkGetTimerValue(&hTimer);
//   printf("CPU Convolve: %.5fmsec\n",(float)sdkGetTimerValue(&hTimer));

//   // cudaMemcpy(B, B_gpu, w * h * sizeof(float), cudaMemcpyDeviceToHost);

//   // for (int i = 0; i < w; i++) {
//   //   for (int j = 0; j < h; j++) {
//   //     printf("%.2f,", A[i*h+j]);
//   //   }
//   //   printf("\n");
//   // }
//   // printf("\n");
//   // for (int i = 0; i < w; i++) {  
//   //   for (int j = 0; j < h; j++) {
//   //     printf("%.2f,", B[i*h+j]);
//   //   }
//   //   printf("\n");
//   // }
//   cudaFree(A_gpu);
//   cudaFree(B_gpu);
//   cudaFree(H_gpu);
//   free(B_cpu);
//   free(B);
//   free(A);
//   free(H);
//   return 0;
// }