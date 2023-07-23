#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

__global__ void add_kernel(int *c, const int *a, const int *b) {
  int idx = threadIdx.x;
  c[idx] = a[idx] + b[idx];
}

int main()
{
  // 1. Set host memory
  const int SIZE = 5;
  const int BYTES = SIZE * sizeof(int);
  const int a[SIZE] = {1, 2, 3, 4, 5};
  const int b[SIZE] = {10, 20, 30, 40, 50};
  int c[SIZE] = {0,};

  // 2. Set Pointers for device memory
  int* dev_a = nullptr;
  int* dev_b = nullptr;
  int* dev_c = nullptr;


  // 3. Allocate device memory
  if (cudaMalloc((void**)&dev_a, BYTES) != cudaSuccess) {
    printf("cudaMalloc dev_a failed\n");
    return 1;
  }
  if (cudaMalloc((void**)&dev_b, BYTES) != cudaSuccess) {
    printf("cudaMalloc dev_b failed\n");
    return 1;
  }
  if (cudaMalloc((void**)&dev_c, BYTES) != cudaSuccess) {
    printf("cudaMalloc dev_c failed\n");
    return 1;
  }

  // 4. Copy source host memory to device memory
  if (cudaMemcpy(dev_a, a, BYTES, cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("cudaMemcpy host to device failed\n");
    return 1;
  }
  if (cudaMemcpy(dev_b, b, BYTES, cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("cudaMemcpy host to device failed\n");
    return 1;
  }

  // 5. Launch kernel & Check error 
  add_kernel<<<1, SIZE>>>(dev_c, dev_a, dev_b);
#if defined(__linux__)
  cudaDeviceSynchronize();
#endif
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("kernel launch failed: %s\n", cudaGetErrorString(err));
    return 1;
  } else {
    printf("kernel launch success!\n");
  }

  //6. Copy result device memory to host memory
  if (cudaMemcpy(c, dev_c, BYTES, cudaMemcpyDeviceToHost) != cudaSuccess) {
    printf("cudaMemcpy device to host failed\n");
    return 1;
  }

  // 7. Free device memory
  if (cudaFree(dev_a) != cudaSuccess) {
    printf("cudaFree dev_a failed\n");
    return 1;
  }
  if (cudaFree(dev_b) != cudaSuccess) {
    printf("cudaFree dev_b failed\n");
    return 1;
  }
  if (cudaFree(dev_c) != cudaSuccess) {
    printf("cudaFree dev_c failed\n");
    return 1;
  }

  // 8. Print result
  printf("c[i] = a[i] + b[i]\n");
  printf("%d %d %d %d %d\n", c[0], c[1], c[2], c[3], c[4]);

  return 0;
}