#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// const unsigned int SIZE = 1 * 1024 * 1024;
const unsigned int SIZE = 1024;
__constant__ float dev_a = 1.234f;
__device__ float d_vecX[SIZE];
__device__ float d_vecY[SIZE];
__device__ float d_vecZ[SIZE];

void setRandomData(float data[], const int size, const float bound=static_cast<float>(1000)) {
  int bnd = static_cast<int>(bound);
  for (int i = 0; i < size; i++) {
    data[i] = (rand() % bnd) / bound;
  }
}

float getSum(const float data[], const int size) {
  float sum = 0.f;
  for (int i = 0; i < size; i++) {
    sum += data[i];
  }
  return sum;
}

__global__ void kernelVecAddGPU(const unsigned int size) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    d_vecZ[id] = dev_a * d_vecX[id] + d_vecY[id];
  }
}

int main()
{
  std::chrono::steady_clock::time_point total_begin, total_end;
  std::chrono::microseconds total_elapsed_time;
  total_begin = std::chrono::steady_clock::now();
  float saxpy_a = 1.234f;

  float vecX[SIZE];
  float vecY[SIZE];
  float vecZ[SIZE];

  srand(0);
  setRandomData(vecX, SIZE);
  setRandomData(vecY, SIZE);

  cudaMemcpyToSymbol(d_vecX, vecX, sizeof(vecX));
  cudaMemcpyToSymbol(d_vecY, vecY, sizeof(vecY));
  cudaMemcpyToSymbol(d_vecZ, vecZ, sizeof(vecZ));

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  kernelVecAddGPU<<<(SIZE + 1023) / 1024, 1024>>>(SIZE);
  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::microseconds time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  std::cout << "Time: " << time.count() << " us" << std::endl;

  cudaMemcpyFromSymbol(vecZ, d_vecZ, sizeof(vecZ));

  std::cout << "SIZE:\t" << SIZE << std::endl;
  std::cout << "a:\t" << saxpy_a << std::endl;
  std::cout.precision(10);
  std::cout << "Sum(vecX):\t" << getSum(vecX, SIZE) << std::endl;
  std::cout << "Sum(vecY):\t" << getSum(vecY, SIZE) << std::endl;
  std::cout << "Sum(vecZ):\t" << getSum(vecZ, SIZE) << std::endl;
  std::cout << "diff:\t" << getSum(vecZ, SIZE) - getSum(vecX, SIZE) * saxpy_a - getSum(vecY, SIZE) << std::endl;
  std::cout << "VecX:\t" << vecX[0] << ", " << vecX[1] << ", " << vecX[2] << ", " << vecX[3] << std::endl;
  std::cout << "VecY:\t" << vecY[0] << ", " << vecY[1] << ", " << vecY[2] << ", " << vecY[3] << std::endl;
  std::cout << "VecZ:\t" << vecZ[0] << ", " << vecZ[1] << ", " << vecZ[2] << ", " << vecZ[3] << std::endl;

  cudaFree(d_vecX);
  cudaFree(d_vecY);
  cudaFree(d_vecZ);

  total_end = std::chrono::steady_clock::now();
  total_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_begin);
  std::cout << "Total Elapsed time: " << static_cast<float>(total_elapsed_time.count()) / 1000 << "ms" << std::endl;

  return 0;
}