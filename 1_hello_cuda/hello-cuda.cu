#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

__global__ void hello() {
  printf("Hello CUDA! %d\n", threadIdx.x);
}

int main() {
  hello<<<2, 5>>>();

#if defined(__linux__)
  cudaDeviceSynchronize();
#endif

  return 0;
}