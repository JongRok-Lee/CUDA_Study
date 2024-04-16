#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#if defined(NDEBUG)
#define CUDA_CHECK_ERROR() 0
#else
#define CUDA_CHECK_ERROR() \
  do { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
      printf("CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1); \
    } \
  } while (0)
#endif

__global__ void add_kernel(float *b, const float *a) {
  int idx = threadIdx.x;
  b[idx] = a[idx] + 1.0f;
}

int main()
{
  // 1. Set host memory
  const int SIZE = 8;
  const int BYTES = SIZE * sizeof(float);
  const float a[SIZE] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float b[SIZE] = {0.0f,};

  // 2. Set Pointers for device memory
  float* dev_a = nullptr;
  float* dev_b = nullptr;


  // 3. Allocate device memory
  if (cudaMalloc((void**)&dev_a, BYTES) != cudaSuccess) {
    printf("cudaMalloc dev_a failed\n");
    return 1;
  }
  if (cudaMalloc((void**)&dev_b, BYTES) != cudaSuccess) {
    printf("cudaMalloc dev_b failed\n");
    return 1;
  }

  // 4. Copy source host memory to device memory
  if (cudaMemcpy(dev_a, a, BYTES, cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("cudaMemcpy host to device failed\n");
    return 1;
  }

  // 5. Launch kernel & Check error 
  add_kernel<<<1, SIZE>>>(dev_b, dev_a);
#if defined(__linux__)
  cudaDeviceSynchronize();
#endif
  CUDA_CHECK_ERROR();

  //6. Copy result device memory to host memory
  if (cudaMemcpy(b, dev_b, BYTES, cudaMemcpyDeviceToHost) != cudaSuccess) {
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

  // 8. Print result
  printf("b[i] = a[i] + 1.0\n");
  printf("%f %f %f %f %f %f %f %f\n", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);

  // 9. Raise error
  cudaMemcpy(b, dev_b, BYTES, cudaMemcpyDeviceToDevice);
  CUDA_CHECK_ERROR();

  return 0;
}