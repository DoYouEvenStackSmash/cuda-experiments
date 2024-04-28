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


__global__ void find_min(double *A, double *vB, double *Aprime, double thres, double lb, int width, int height, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((int)(col/width)== (height - lb)) {
    int counter = (col) / width;
    for (int j = 0; j < counter; j++) {
      if (col - j * width < 0)break;
      if (A[col - j * width] < thres) {
        vB[col % width] = Aprime[col  - (j-1) * width];
        break;
      }
    }
  }
}

void find_min_wrap(uint64_t ptr1, uint64_t ptr2, uint64_t back_ptr, double thres, int lb, int width, int height) {
  find_min<<<width, height>>>((double*)ptr1, (double*)ptr2, (double*)back_ptr, thres, lb,width, height, width*height);
}

__global__ void prewarp(double* A, double* B, double mid, double max_px, double theta_max, double cam_height, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < buflen) {
    double theta_oc = (col - mid) / max_px * theta_max;
    double range_oc = cam_height / (tan(asin(cam_height / A[col])));
    B[col] = range_oc / cos(theta_oc);
  }
}

void prewarp_wrap(uint64_t ptr1, uint64_t ptr2, double mid, double max_px, double theta_max, double cam_height, int buflen) {
  prewarp<<<20,32>>>((double*)ptr1, (double*)ptr2, mid, max_px, theta_max, cam_height, buflen);
}

__global__ void conv(double* A,double* B, double* H, int hlen, int ylen,int width, int height, int lb, int span) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
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

///
// Scalar operations
///
__global__ void scalar_divide(double* a, double val, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < buflen) {
    a[col] = a[col] / val;
  }
}

void scalar_divide_wrap(uint64_t ptr, double val, double width, double height) {
  scalar_divide<<<width, height>>>((double*)ptr, val, width*height);
}

__global__ void invert(double *a, double val, int buflen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < buflen) {
    a[col] = val / (a[col]+1);
  }
}

void invert_wrap(uint64_t ptr,double val, double width, double height) {
  invert<<<width, height>>>((double*)ptr, val, width*height);
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

////
// Memory management functions
////
void move_to_gpu_addr(uint64_t ptr, py::array_t<double> arr1,int sz) {
  auto buf1 = arr1.request();
  double *ptr1 = (double *)buf1.ptr;
  double *buf1_gpu = (double*)ptr;

  int buflen = sz;
  cudaMemcpy(buf1_gpu, ptr1, buflen * sizeof(double),cudaMemcpyHostToDevice);
}

void move_to_cpu(uint64_t ptr, py::array_t<double> arr1) {
  auto buf1 = arr1.request();
  double *ptr1 = (double *)buf1.ptr;
  double *buf1_gpu = (double*)ptr;
  int buflen = buf1.shape[0];
  // don't do this
  //memcpy((void**)&(buf1.ptr), (void**)&buf1_gpu, sizeof(double*));
  cudaMemcpy(ptr1, buf1_gpu,buflen * sizeof(double),cudaMemcpyDeviceToHost);
}

void cuda_conv(py::array_t<double> arr1, py::array_t<double> arr2, py::array_t<double> filter) {

  // Get pointers to the data
  auto buf1 = arr1.request(), buf2 = arr2.request(), buf3 = filter.request();
  double *ptr1 = (double *)buf1.ptr, *ptr2 = (double *)buf2.ptr, *ptr3 = (double *)buf3.ptr;

  conv_wrap(ptr1, ptr2, ptr3, buf3.shape[0], buf1.shape[0]*buf1.shape[1], buf1.shape[0], buf1.shape[1], 0, 0);

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

__global__ void pairwise_min(double* vals, double* vals_ages, double* inc_vals, int ylen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < ylen) {
    if (vals[col] > inc_vals[col]) {
      vals[col] = (double)inc_vals[col];
      vals_ages[col] = 0;
    }
  }
}

__global__ void pairwise_age(double* vals, double* vals_ages, double*inc_vals, int max_age, int ylen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < ylen) {
    if (vals_ages[col] && vals_ages[col] > max_age) {
      vals[col] = (double)inc_vals[col];
      vals_ages[col] = 0;
    }
    vals_ages[col]++;
  }
}

void sliding_window(uint64_t vals, uint64_t vals_ages, uint64_t inc_vals, int max_age,int ylen) {
  pairwise_min<<<640,416>>>((double*)vals, (double*)vals_ages, (double*) inc_vals, ylen);
  pairwise_age<<<640,416>>>((double*)vals, (double*)vals_ages, (double*) inc_vals, max_age, ylen);
  cudaDeviceSynchronize();
}
__global__ void minimize(double* vals, double* vals_ages, double* inc_vals, int max_age,int ylen) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < ylen) {
    if (vals_ages[col] < max_age && vals[col] < inc_vals[col]) {
      vals_ages[col]++;
    }
    else {
      vals[col] = (double)inc_vals[col];
      vals_ages[col] = 0;
    }
    // vals[col] = vals[col] <= inc_vals[col] ? vals[col] : (double)inc_vals[col];
    
    
  }
}

void minimize_wrap(double* vals, double* val_ages, double* new_vals, int lifetime, int width, int height, int buflen){
  int threads = 1024;
  int off_t = 0;
  if (threads == 1024) {
    off_t = buflen/1024;
  }
  // int blocks = ceil(ylen / min(height, threads))
  minimize<<<width,height>>>(vals,val_ages,new_vals, lifetime, width*height);
}

void direct_minimize_wrap(uint64_t vals,uint64_t ages,uint64_t new_vals, int lifetime,int width, int height, int buflen) {
  minimize_wrap((double*)vals, (double*)ages, (double*)new_vals, lifetime, width, height, buflen);
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

uint64_t a_number() {
  return rand();
}

// Binding code
PYBIND11_MODULE(numpy_multiply, m) {
    m.def("a_number", a_number, "return a number");
    m.def("warmup", warmup, "warmup the gpu");
    m.def("move_to_gpu", move_to_gpu, "move an array to the gpu");
    m.def("move_to_cpu", move_to_cpu, "move an array to the cpu");
    m.def("free_gpu", free_gpu, "free gpu memory");
    m.def("direct_conv_wrap",direct_conv_wrap,"direct_conv wrapper");
    m.def("move_to_gpu_addr", move_to_gpu_addr, "moves to an address on gpu");
    m.def("minimize_wrap", minimize_wrap, "wrapper for element wise minimization");
    m.def("direct_minimize_wrap", direct_minimize_wrap, "direct wrapper for minimize_wrap");
    m.def("scalar_divide_wrap", scalar_divide_wrap,"wrapper for scalar division");
    m.def("find_min_wrap", find_min_wrap, "wrapper for finding obstacle min");
    m.def("invert_wrap", invert_wrap, "inverted division");
    m.def("prewarp_wrap", prewarp_wrap, "prewarp wrapper function");
    m.def("sliding_window", sliding_window, "sliding window function");
}
