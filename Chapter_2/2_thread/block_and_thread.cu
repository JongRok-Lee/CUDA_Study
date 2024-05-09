#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

__global__ void print1DIndex() {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%d x %d + %d = %d\n", blockIdx.x, blockDim.x, threadIdx.x, id);
}

__global__ void print2DIndex() {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  int id = gy * gridDim.x + gx;

  printf("gx: %d x %d + %d = %d\t\
          gy: %d x %d + %d = %d\t\
          id: %d x %d + %d = %d\n", blockIdx.x, blockDim.x, threadIdx.x, gx,\
                                    blockIdx.y, blockDim.y, threadIdx.y, gy,\
                                    gy, gridDim.x, gx, id);
}

__global__ void print3DIndex() {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  int gz = blockIdx.z * blockDim.z + threadIdx.z;
  int id = gz * gridDim.x * gridDim.y + gy * gridDim.x + gx;

  printf("gx: %d x %d + %d = %d\t\
          gy: %d x %d + %d = %d\t\
          gz: %d x %d + %d = %d\t\
          id: %d x %d x %d + %d x %d + %d = %d\n", blockIdx.x, blockDim.x, threadIdx.x, gx,\
                                                   blockIdx.y, blockDim.y, threadIdx.y, gy,\
                                                   blockIdx.z, blockDim.z, threadIdx.z, gz,\
                                                   gz, gridDim.x, gridDim.y, gy, gridDim.x, gx, id);
}

int main() {
  print1DIndex<<<dim3(6), dim3(4)>>>();
  cudaDeviceSynchronize();
  printf("\n");

  print2DIndex<<<dim3(3, 5), dim3(4, 3)>>>();
  cudaDeviceSynchronize();
  printf("\n");
  
  print3DIndex<<<dim3(3, 5, 2), dim3(4, 3, 2)>>>();
  cudaDeviceSynchronize();
  printf("\n");

  return 0;
}