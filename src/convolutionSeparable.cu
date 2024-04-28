#include <assert.h>
#include <helper_cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "convolutionSeparable_common.h"
////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(float *h_Result, float *h_Data,
                                  float *h_Kernel, int imageW, int imageH,
                                  int kernelR);

extern "C" void convolutionColumnCPU(float *h_Result, float *h_Data,
                                     float *h_Kernel, int imageW, int imageH,
                                     int kernelR);

// extern "C" void D_Conv_i(float *out,float *in, unsigned int numElements);

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////

// __constant__ float fl1[SIZE_FL1];
__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float *h_Kernel) {
  cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}

__global__ void D_Conv_i(float *out,float *in, unsigned int numElements)
{

    unsigned long int i = blockIdx.x + threadIdx.x;
    int k = 0; // 1
    printf("%d",k);
    if (i < numElements)
    {
        float s = 0;
        for (int j = 0; j < KERNEL_LENGTH; j++) {
          if  (i - j >= 0 && i - j < numElements) s += 1 * in[i - j];
        }
        out[i] += s;
    }
    
}
void deconvi(dim3 grid, dim3 threads, float* h_OutputGPU, float* h_Input, unsigned int ops) {
  D_Conv_i<<<grid, threads>>>(h_OutputGPU,h_Input,ops);
}