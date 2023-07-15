#include <stdio.h>

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