#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// #include <stdio.h>
// #include <stdlib.h>
#include <cuda_runtime.h>
// // Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
namespace py = pybind11;


__global__ void conv(double* A,double* B, double* H, int hlen, int ylen,int width, int height, int lb, int span) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // int row = blockIdx.y * blockDim.y + threadIdx.y;
  int i = col;
  if (col < ylen && col >= lb * width) {
    double sum = 0.0f;
    for (int j = 0; j < hlen; j++) {
      if (i-j*width < 0)
        break;
      sum = sum + (double) H[j] * A[i - j * width];
    }
    B[i] = sum;
  }
}

void conv_wrap(double* A,double *B, double *H, int hlen, int ylen,int width, int height,int lb,int span) {
  int threads = min(1024,(height + height%32));
  int off_t = 0;
  if (threads == 1024) {
    off_t = height - threads;
  }

  // int blocks = ceil(ylen / min(height, threads))
  conv<<<(width+off_t+width%32),min(1024,(height + height%32))>>>(A,B,H, hlen, ylen,width,height,lb,span);
}

uint64_t move_to_gpu(py::array_t<double> arr1) {
  auto buf1 = arr1.request();
  double *ptr1 = (double *)buf1.ptr;
  double *buf1_gpu;

  int buflen = buf1.shape[0];
  
  cudaMalloc((void**)&buf1_gpu, buflen * sizeof(double));
  cudaMemcpy(buf1_gpu, ptr1, buflen * sizeof(double),cudaMemcpyHostToDevice);
  
  return (uint64_t)buf1_gpu;
}

void move_to_gpu_addr(uint64_t ptr, py::array_t<double> arr1) {
  auto buf1 = arr1.request();
  double *ptr1 = (double *)buf1.ptr;
  double *buf1_gpu = (double*)ptr;

  int buflen = buf1.shape[0];
  
  // cudaMalloc((void**)&buf1_gpu, buflen * sizeof(double));
  cudaMemcpy(buf1_gpu, ptr1, buflen * sizeof(double),cudaMemcpyHostToDevice);
}

void move_to_cpu(uint64_t ptr, py::array_t<double> arr1) {
  auto buf1 = arr1.request();
  double *ptr1 = (double *)buf1.ptr;
  double *buf1_gpu = (double*)ptr;
  int buflen = buf1.shape[0];
  cudaMemcpy(ptr1, buf1_gpu,buflen * sizeof(double),cudaMemcpyDeviceToHost);
  // return 
}

py::array_t<double> cuda_conv(py::array_t<double> arr1, py::array_t<double> arr2, py::array_t<double> filter) {

  // Get pointers to the data
  auto buf1 = arr1.request(), buf2 = arr2.request(), buf3 = filter.request();
  double *ptr1 = (double *)buf1.ptr, *ptr2 = (double *)buf2.ptr, *ptr3 = (double *)buf3.ptr;

  // Create a new NumPy array to store the result
  // py::array_t<double> result({buf1.shape[0]*buf1.shape[1]});
  // auto buf3 = result.request();
  // double *ptr3 = (double *)buf3.ptr;
  
  conv_wrap(ptr1, ptr2, ptr3, buf3.shape[0], buf1.shape[0]*buf1.shape[1], buf1.shape[0], buf1.shape[1], 0, 0);

  // Perform element-wise multiplication
  // return result;
}

void direct_conv_wrap(uint64_t ptr1, uint64_t ptr2, uint64_t filter_ptr, int hlen,int width, int height, int buflen) {
  conv_wrap((double*)ptr1, (double*)ptr2, (double*)filter_ptr, hlen, buflen, width, height, 0, 0);

}

__global__ void c_multiply(double* buf1, double* buf2,int bufcount, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < buflen) {
    buf1[col] = buf1[col] * buf2[col];
  }
}

__global__ void minimize(double* vals, double* vals_ages, double* inc_vals, int max_age, int ylen) {
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
void minimize_wrap(double* vals, double* val_ages, double* new_vals, int buflen){
  int threads = 1024;
  int off_t = 0;
  if (threads == 1024) {
    off_t = buflen/1024;
  }
  // int blocks = ceil(ylen / min(height, threads))
  minimize<<<ceil(off_t),threads>>>(vals,val_ages,new_vals, 5, buflen);
}

void direct_minimize_wrap(uint64_t vals,uint64_t ages,uint64_t new_vals, int buflen) {
  minimize_wrap((double*) vals, (double*) ages, (double*) new_vals, buflen);
}

uint64_t warmup(int size) {
  int devID = gpuGetMaxGflopsDeviceId();
  checkCudaErrors(cudaSetDevice(devID));
  int major = 0, minor = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
         devID, _ConvertSMVer2ArchName(major, minor), major, minor);
  double *buf1_gpu;
  cudaMalloc((void**)&buf1_gpu, size * sizeof(double));
  return (uint64_t)buf1_gpu;
}

void free_gpu(uint64_t ptr) {
  cudaFree((double*)ptr);
}

void cuda_multiply(double* ptr3, double* ptr1, double* ptr2, int bufcount, int buflen) {
  int devID = gpuGetMaxGflopsDeviceId();
  checkCudaErrors(cudaSetDevice(devID));
  int major = 0, minor = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
         devID, _ConvertSMVer2ArchName(major, minor), major, minor);
  double *buf1_gpu;
  double *buf2_gpu;
  // double *buf3_gpu;
  cudaMalloc((void**)&buf1_gpu, buflen * sizeof(double));
  cudaMemcpy(buf1_gpu, ptr1, buflen * sizeof(double),cudaMemcpyHostToDevice);
  cudaMalloc((void**)&buf2_gpu, buflen * sizeof(double));
  cudaMemcpy(buf2_gpu, ptr2, buflen * sizeof(double),cudaMemcpyHostToDevice);
  // cudaMalloc((void**) &buf3_gpu, buflen*sizeof(double));

  /*
  cudaStream_t stream;
  checkCudaErrors(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start, stream));
*/
  // dim3 block_shape(32,32);

  // dim3 grid_shape(max(1.0, ceil((double)buflen / (double)block_shape.x)),max(1.0, ceil((double)buflen / (double)block_shape.y)));

  c_multiply<<<bufcount, buflen>>>(buf1_gpu, buf2_gpu, bufcount, buflen);
  /*
  checkCudaErrors(cudaEventRecord(stop, stream));
  
  double msecTotal=0.0f;

  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  int iter = 1;
  double msecPerMatrixMul = msecTotal/1e9;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(buflen) *
                            static_cast<double>(1) * static_cast<double>(20) * iter;
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, 32);
  */
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

    // Get pointers to the data
    auto buf1 = arr1.request(), buf2 = arr2.request();
    double *ptr1 = (double *)buf1.ptr, *ptr2 = (double *)buf2.ptr;

    // Create a new NumPy array to store the result
    // py::array_t<double> result({buf1.shape[0]*buf1.shape[1]});
    // auto buf3 = result.request();
    // double *ptr3 = (double *)buf3.ptr;
    
    cuda_multiply(ptr1, ptr1, ptr2, 1, buf1.shape[0]*buf1.shape[1]);

    // Perform element-wise multiplication
    // return result;
}

uint64_t a_number() {
  return rand();
}

// Binding code
PYBIND11_MODULE(numpy_multiply, m) {
    m.def("array_multiply", array_multiply, "Multiply two numpy arrays");
    m.def("a_number", a_number, "return a number");
    m.def("warmup", warmup, "warmup the gpu");
    m.def("move_to_gpu", move_to_gpu, "move an array to the gpu");
    m.def("move_to_cpu", move_to_cpu, "move an array to the cpu");
    m.def("free_gpu", free_gpu, "free gpu memory");
    m.def("direct_conv_wrap",direct_conv_wrap,"direct_conv wrapper");
    m.def("move_to_gpu_addr", move_to_gpu_addr, "moves to an address on gpu");
    m.def("minimize_wrap", minimize_wrap, "wrapper for element wise minimization");
    m.def("direct_minimize_wrap", direct_minimize_wrap, "direct wrapper for minimize_wrap");

}
