// #include "kernels.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <thread>

std::vector<double> generateGaussianKernel(int n, double sigma) {
  std::vector<double> kernel(n);
  double sum = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = i - (n - 1) / 2;
    kernel[i] = exp(-(x * x) / (2 * sigma * sigma)) / (sqrt(2 * M_PI) * sigma);
    sum += kernel[i];
  }
  
  for (int i = 0; i < n; ++i) {
    kernel[i] /= sum;
  }

  return kernel;
}

int main() {
  std::vector<double> foo = generateGaussianKernel(9,1);
  for (auto a : foo)
    std::cout << a << ',';
  return 0;

}