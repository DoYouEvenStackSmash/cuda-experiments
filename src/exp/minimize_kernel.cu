#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <fstream>
#include <string>
__global__ void minimize(float* vals, float* vals_ages, float* inc_vals, int max_age, int ylen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < ylen) {
    if (vals_ages[col] >= max_age) {
      vals[col] = inc_vals[col];
      vals_ages[col] = 0;
    }
    vals[col] = vals[col] <= inc_vals[col] ? vals[col] : inc_vals[col];
    vals_ages[col]++;
  }
}

int main(int argc, char** argv) {
  findCudaDevice(argc, (const char **)argv);
  float *vals, *val_ages, *inc_vals;
  float *vals_gpu, *val_ages_gpu, *inc_vals_gpu;
  int cols = 5;
  int rows = 4;
  srand(2006);
  vals = (float *)malloc(rows * cols * sizeof(float));
  val_ages = (float *)malloc(rows * cols * sizeof(float));
  inc_vals = (float *)malloc(rows * cols * sizeof(float));
  for (int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      vals[i * cols + j] = rand() / (float)RAND_MAX;
      val_ages[i * cols + j] = 0;
      inc_vals[i * cols + j] = rand() / (float)RAND_MAX;
    }
  }
  for (int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      printf("%.2f\t",vals[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
  cudaMalloc((void **) &vals_gpu,rows * cols * sizeof(float));

  cudaMalloc((void **) &val_ages_gpu,rows * cols * sizeof(float));

  cudaMalloc((void **) &inc_vals_gpu,rows * cols * sizeof(float));

  cudaMemcpy(vals_gpu,vals,rows * cols * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(val_ages_gpu,val_ages,rows * cols * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(inc_vals_gpu,inc_vals,rows * cols * sizeof(float),cudaMemcpyHostToDevice);
  int ylen = rows * cols;
  minimize<<<rows,cols>>>(vals_gpu, val_ages_gpu, inc_vals_gpu, 5, ylen);

  cudaMemcpy(vals,vals_gpu,rows * cols * sizeof(float),cudaMemcpyDeviceToHost);
  

  for (int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      printf("%.2f\t",inc_vals[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
  for (int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      printf("%.2f\t",vals[i * cols + j]);
    }
    printf("\n");
  }
  free(vals);
  free(val_ages);
  free(inc_vals);
  cudaFree(vals_gpu);
  cudaFree(val_ages_gpu);
  cudaFree(inc_vals_gpu);
  return 0;
  
}