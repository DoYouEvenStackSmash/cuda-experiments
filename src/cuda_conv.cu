#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <fstream>
#include <string>
#include "common_kernels.h"

void cpu_convolve(float *A, float *B, float *H, int hlen, int ylen, int width, int height,int lb=0,int span=0) {
  for (int r = 0; r < width; r++) {
    for (int i = 0; i < height; i++) {
      float sum = 0.0f;
      for (int j = 0; j < hlen; j++) {
        if (i - j < 0) break;
        sum = sum + H[j] * A[(i-j)*width];
      }
      B[r + i * width] += sum;
    }
  }
}

float dataloader(char* filename, float** buf) {
  // Open the binary file for reading
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
      std::cerr << "Failed to open file." << std::endl;
      return 1;
  }

  // Determine the size of the file
  file.seekg(0, std::ios::end);
  std::streampos fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // Calculate the size of each array
  std::streampos arraySize = fileSize / 600;
  float* buffer = (float *)malloc(fileSize * sizeof(float));
  // Read the file into 600 arrays
  std::vector<std::vector<char>> arrays(600);
  char* cbuf = (char*)malloc((int)arraySize*sizeof(char));
  for (int i = 0; i < 600; ++i) {
      // Resize the vector to hold the data for one array
      // std::cout << arraySize << std::endl;
      // arrays[i].resize(arraySize);
      // Read data into the vector
      file.read(cbuf,arraySize);
      for (int x = 0; x < arraySize; x++)
        buffer[i * arraySize+x] = (float)cbuf[x];
      // file.read(arrays[i].data(), arraySize);
      // Check for errors
      if (file.bad()) {
          std::cerr << "Error reading file." << std::endl;
          return 1;
      }
  }
  free(cbuf);

  // Close the file
  file.close();
  *buf = buffer;
  return fileSize;

}

int datawriter(float* buffer, float bufferSize) {
      // Open the binary file for writing
  // Open the binary file for writing
  std::ofstream file("output.bin", std::ios::binary);
  if (!file.is_open()) {
      std::cerr << "Failed to open file." << std::endl;
      free(buffer); // Clean up
      return 1;
  }

  // Write the buffer to the file
  file.write(reinterpret_cast<const char*>(buffer), bufferSize * sizeof(float));
  if (!file.good()) {
      std::cerr << "Error writing to file." << std::endl;
      free(buffer); // Clean up
      return 1;
  }

  // Close the file
  file.close();

  std::cout << "File written successfully." << std::endl;

  // Clean up
  free(buffer);

  return 0;

}

int main(int argc, char** argv) {
  float* databuf = NULL;
  float bytes = dataloader(argv[1],&databuf);
  int count = bytes / (640*400);
  printf("%.2f",bytes);
  // datawriter(databuf,bytes);
  // free(databuf);
  // return 0;
  // free(databuf);
  // return 0;
  StopWatchInterface *hTimer = NULL;
  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  
  findCudaDevice(argc, (const char **)argv);
  sdkCreateTimer(&hTimer);

  float *A, *B, *H;
  float *A_gpu, *B_gpu,*H_gpu;
  float *blur, *deriv, *mean;
  float *blur_gpu, *deriv_gpu, *mean_gpu;
  float *B_cpu;
  // float *databuf;
  int w = 640;
  int h = 400;
  int ylen = w * h;
  
  // float *gaussian;
  int lb = 2;
  int span = h;

  float gaussian[9] = {0.000133831,0.00443186,0.0539911,0.241971,0.398943,0.241971,0.0539911,0.00443186,0.000133831};
  int k = 9;

  blur = (float *)malloc(k * sizeof(float));
  for (int i = 0; i < k; i++)
    blur[i] = gaussian[i];
  cudaMalloc((void **) &blur_gpu, k*sizeof(float));
  cudaMemcpy(blur_gpu, blur,k*sizeof(float), cudaMemcpyHostToDevice);

  mean = (float *)malloc(k * sizeof(float));
  for (int i = 0; i < k; i++)
    mean[i] = (float)1/k;
  cudaMalloc((void **) &mean_gpu, k*sizeof(float));
  cudaMemcpy(mean_gpu, mean,k*sizeof(float), cudaMemcpyHostToDevice);

  deriv = (float *)malloc(2 * sizeof(float));
  deriv[0] = -1.0;
  deriv[1] = 1.0;
  cudaMalloc((void **) &deriv_gpu, 2*sizeof(float));
  cudaMemcpy(deriv_gpu,deriv,2*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(mean,mean_gpu,k*sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(blur,blur_gpu,k*sizeof(float), cudaMemcpyDeviceToHost);
//{
  cudaMalloc((void **) &A_gpu, w * h * sizeof(float));
  cudaMalloc((void **) &B_gpu, w * h * sizeof(float));
  B = (float *)malloc(w * h * sizeof(float));
  cudaMemcpy(B_gpu, B, w * h* sizeof(float), cudaMemcpyHostToDevice);

  
  cudaStream_t stream;

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float* orig_A = A_gpu;
  float* orig_B = B_gpu;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaStreamSynchronize(stream));
  checkCudaErrors(cudaEventRecord(start, stream));
  float* output_buffer = (float *)malloc(bytes * sizeof(float));
  for (int i = 0; i < count; i++) {
    cudaMemcpy(A_gpu, &(databuf[i * ylen]), w * h * sizeof(float), cudaMemcpyHostToDevice);

    float* stages[3] = {blur_gpu,deriv_gpu,mean_gpu};
    int flens[3] = {9,2,9};
    int windowing[3] = {0,0,0};

    float* hold = NULL;
    for (int j = 0; j < 3; j++) {
      lb = windowing[j];
      int flen = flens[j];
      float* Hgpu = stages[j];
      conv_wrap(A_gpu,B_gpu,Hgpu, flen, ylen,w,h,lb,span);
      if (j < 2) {
        hold = A_gpu;
        A_gpu = B_gpu;
        B_gpu = hold;
      }
    }

    cudaMemcpy(&output_buffer[i * ylen], B_gpu, w * h * sizeof(float), cudaMemcpyDeviceToHost);
  }
  datawriter(output_buffer, bytes);
  // free(output_buffer);
  free(databuf);
    // const_conv_wrap(A_gpu,B_gpu,ylen,w,h);
  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

//  Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));
  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  // Compute and print the performance
  int iter = 1;
  float msecPerMatrixMul = msecTotal/(iter*20);
  double flopsPerMatrixMul = 2.0 * static_cast<double>(w) *
                            static_cast<double>(h) * static_cast<double>(20) * iter;
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, 32);
  // cudaMemcpy(A, orig_A, w * h * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(deriv,deriv_gpu,2*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(mean,mean_gpu,k*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(blur,blur_gpu,k*sizeof(float), cudaMemcpyDeviceToHost);

  // for (int i = 0; i < row; i++) {
  //   for (int j = 0; j < col; j++) {
  //     printf("%.2f\t", A[i*col+j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");
  // for (int i = 0; i < row; i++) {  
  //   for (int j = 0; j < col; j++) {
  //     printf("%.2f\t", B[i*col+j]);
  //   }
  //   printf("\n");
  // }
  cudaFree(orig_A);
  // free(A);
  cudaFree(orig_B);
  // free(B);
  // cudaFree(H_gpu);
  
  cudaFree(blur_gpu);
  free(blur);

  cudaFree(deriv_gpu);
  free(deriv);
  
  cudaFree(mean_gpu);
  free(mean);
  // free(B_cpu);
  
  // free(H);
  return 0;
}