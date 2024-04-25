#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>


__global__ void hello_cuda() {
  printf("Hello cuda - ");
  printf("BlockIdx X: %d, BlockIdx Y: %d, ThreadIdx X: %d, ThreadIdx Y: %d\n",
    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main(int argc, char** argv) {
  hello_cuda<<<2,2>>>();
  cudaDeviceSynchronize();
  return 0;
}