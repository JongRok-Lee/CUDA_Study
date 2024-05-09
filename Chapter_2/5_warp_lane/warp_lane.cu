#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

__device__ unsigned int warp_id() {
  unsigned int ret;
  asm("mov.u32 %0, %warpid;" : "=r"(ret));
  return ret;
}

__device__ unsigned int lane_id() {
  unsigned int ret;
  asm("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

__global__ void kernelWarpLane() {
  unsigned int warpId = warp_id();
  unsigned int laneId = lane_id();

  if(warpId == 0) {
    printf("Warp ID: %d, Lane ID: %d, Block: (%d,%d), Thread: (%d,%d)\n", warpId, laneId, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
  }
}

int main() {
  dim3 dimGrid(2, 2);
  dim3 dimBlock(16, 16);
  kernelWarpLane<<<dimGrid, dimBlock>>>();
  cudaDeviceSynchronize();

  return 0;
}