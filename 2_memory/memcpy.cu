#include <stdio.h>

int main()
{
  // 1. Set host memory
  const int SIZE = 8;
  const int BYTES = SIZE * sizeof(float);
  const float a[SIZE] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  float b[SIZE] = {0.f,};

  // 2. Set pointer
  float *dev_a = nullptr;
  float *dev_b = nullptr;

  // 3. Allocate device memory
  if (cudaMalloc((void**)&dev_a, BYTES) != cudaSuccess) {
    printf("cudaMalloc dev_a failed!\n");
    return 1;
  }
  if (cudaMalloc((void**)&dev_b, BYTES) != cudaSuccess) {
    printf("cudaMalloc dev_b failed!\n");
    return 1;
  }

  // 4. Copy host memory to device
  if (cudaMemcpy(dev_a, a, BYTES, cudaMemcpyHostToDevice) != cudaSuccess) { // dev_a = a
    printf("cudaMemcpy a to dev_a failed!\n");
    return 1;
  }
  if (cudaMemcpy(dev_b, dev_a, BYTES, cudaMemcpyDeviceToDevice) != cudaSuccess) {// dev_b = dev_a
    printf("cudaMemcpy dev_a to dev_b failed!\n");
    return 1;
  }
  if (cudaMemcpy(b, dev_b, BYTES, cudaMemcpyDeviceToHost) != cudaSuccess) { // b = dev_b
    printf("cudaMemcpy dev_b to b failed!\n");
    return 1;
  }

#if defined(__linux__)
  cudaDeviceSynchronize();
#endif

  // 5. Check result
  for (int i = 0; i < SIZE; ++i)
  {
    printf("b[%d]: %.2f\n", i, b[i]);
  }

  // 6. Free device memory
  if (cudaFree(dev_a) != cudaSuccess) {
    printf("cudaFree dev_a failed!\n");
  }
  if (cudaFree(dev_b) != cudaSuccess) {
    printf("cudaFree dev_b failed!\n");
  }

  return 0;
}