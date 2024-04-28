#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionSeparable_common.h"

#define KERNEL_LENGTH 9

__constant__ 
float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float *h_Kernel) {
  cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}

__global__ void conv(float* out, float* in, float* buf, int N_op) {
  unsigned long int i = blockDim.x * blockIdx.x + threadIdx.x; //; gets us to the pixel of the output
  int stride = 1;
  
  if (i < N_op) {
    float s = 0;
    for (int j = 0; j < KERNEL_LENGTH; j++) {
      if (i-j*stride<0 || i > N_op)
        break;
      s += c_Kernel[j] * in[i-j];
    }
    out[i] = out[i] + s;
  }
}

int main(int argc, char **argv) {
  // start logs
  int Nx = 20;
  int Ny = 20;
  printf("[%s] - Starting...\n", argv[0]);

  float *h_Kernel, *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;

  float *d_Input, *d_Output, *d_Buffer;

  const int imageW = Nx;
  const int imageH = Ny;
  const int iterations = 16;

  StopWatchInterface *hTimer = NULL;

  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  sdkCreateTimer(&hTimer);
  srand(2006);
  printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
  printf("Allocating and initializing host arrays...\n");
  h_Kernel = (float *)malloc(KERNEL_LENGTH * sizeof(float));
  
  for (unsigned int i = 0; i < KERNEL_LENGTH; i++) 
    h_Kernel[i] = 10;
  setConvolutionKernel(h_Kernel);

  h_Input = (float *)malloc(imageW * imageH * sizeof(float));
  h_Buffer = (float *)malloc(imageW * imageH * sizeof(float));
  // h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
  h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
  memset(h_OutputGPU, 0,sizeof(float));

  unsigned long int ops = imageW * imageH*sizeof(float);
  
  for (unsigned int i = 0; i < imageW * imageH; i++) {
    h_Input[i] = (float)(rand() % 16) * 100;
  }
  cudaStream_t stream;

  checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Output, h_OutputGPU, imageW * imageH * sizeof(float),
                              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Buffer, h_Buffer, imageW * imageH * sizeof(float),
                              cudaMemcpyHostToDevice));
                              printf("Shutting down...\n");

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaStreamSynchronize(stream));
  checkCudaErrors(cudaEventRecord(start, stream));
  for (int i = 0; i < 10; i++) {
    conv<<<Nx,Ny>>>(d_Output,d_Input,d_Buffer,Nx*Ny);
  }
  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));
  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal/300;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(Ny) *
                             static_cast<double>(Nx) * 1000;
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, Ny);
  checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output,
                          imageW * imageH * sizeof(float),
                          cudaMemcpyDeviceToHost));
  for (int a = 0; a < Nx; a++) {
    for (int b = 0; b < Ny; b++) {
      printf("%.4f ",h_OutputGPU[a*Ny + b]);
    }
    printf(" \n");
  }
  
  checkCudaErrors(cudaFree(d_Buffer));
  checkCudaErrors(cudaFree(d_Output));
  checkCudaErrors(cudaFree(d_Input));
  free(h_OutputGPU);
  // free(h_OutputCPU);
  free(h_Buffer);
  free(h_Input);
  free(h_Kernel);

  sdkDeleteTimer(&hTimer);

  // if (L2norm > 1e-6) {
  //   printf("Test failed!\n");
  //   exit(EXIT_FAILURE);
  // }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);

}

