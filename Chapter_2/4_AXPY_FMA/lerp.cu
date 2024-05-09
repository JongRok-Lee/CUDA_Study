#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

template<typename PREC>
void setRandomData(PREC data[], const int size, const PREC bound=static_cast<PREC>(1000)) {
  int bnd = static_cast<int>(bound);
  for (int i = 0; i < size; i++) {
    data[i] = (rand() % bnd) / bound;
  }
}

template<typename PREC>
PREC getSum(const PREC data[], const int size) {
  PREC sum = static_cast<PREC>(0);
  for (int i = 0; i < size; i++) {
    sum += data[i];
  }
  return sum;
}

template<typename PREC>
__global__ void kernelLERP(PREC vecC[], const PREC vecA[], const PREC vecB[], const int size, const float t) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    vecC[id] = (1 - t) * vecA[id] + t * vecB[id];
  }
}

template<typename PREC>
__global__ void kernelFMALERP(PREC vecC[], const PREC vecA[], const PREC vecB[], const int size, const float t) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    vecC[id] = fmaf(t, vecB[id], fmaf(-t, vecA[id], vecA[id]));
  }
}

int main()
{
  std::chrono::steady_clock::time_point total_begin, total_end;
  std::chrono::microseconds total_elapsed_time;
  total_begin = std::chrono::steady_clock::now();
  const unsigned int SIZE = 256 * 1024 * 1024;
  float lerp_a = 0.234f;
  float *vecX = new float[SIZE];
  float *vecY = new float[SIZE];
  float *vecZ = new float[SIZE];

  srand(0);
  setRandomData(vecX, SIZE);
  setRandomData(vecY, SIZE);

  float *d_vecX, *d_vecY, *d_vecZ;
  cudaMalloc((void**)&d_vecX, SIZE * sizeof(float));
  cudaMalloc((void**)&d_vecY, SIZE * sizeof(float));
  cudaMalloc((void**)&d_vecZ, SIZE * sizeof(float));
  cudaMemcpy(d_vecX, vecX, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vecY, vecY, SIZE * sizeof(float), cudaMemcpyHostToDevice);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  kernelLERP<<<(SIZE + 1023) / 1024, 1024>>>(d_vecZ, d_vecX, d_vecY, SIZE, lerp_a);
  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::microseconds time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  std::cout << "LERF Time: " << time.count() << " us" << std::endl;

  begin = std::chrono::steady_clock::now();
  kernelFMALERP<<<(SIZE + 1023) / 1024, 1024>>>(d_vecZ, d_vecX, d_vecY, SIZE, lerp_a);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  std::cout << "FMA LERP Time: " << time.count() << " us" << std::endl;

  cudaMemcpy(vecZ, d_vecZ, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "SIZE:\t" << SIZE << std::endl;
  std::cout << "a:\t" << lerp_a << std::endl;
  std::cout.precision(10);
  std::cout << "Sum(vecX):\t" << getSum(vecX, SIZE) << std::endl;
  std::cout << "Sum(vecY):\t" << getSum(vecY, SIZE) << std::endl;
  std::cout << "Sum(vecZ):\t" << getSum(vecZ, SIZE) << std::endl;
  std::cout << "VecX:\t" << vecX[0] << ", " << vecX[1] << ", " << vecX[2] << ", " << vecX[3] << std::endl;
  std::cout << "VecY:\t" << vecY[0] << ", " << vecY[1] << ", " << vecY[2] << ", " << vecY[3] << std::endl;
  std::cout << "VecZ:\t" << vecZ[0] << ", " << vecZ[1] << ", " << vecZ[2] << ", " << vecZ[3] << std::endl;

  delete[] vecX, vecY, vecZ;
  cudaFree(d_vecX);
  cudaFree(d_vecY);
  cudaFree(d_vecZ);

  total_end = std::chrono::steady_clock::now();
  total_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_begin);
  std::cout << "Total Elapsed time: " << static_cast<float>(total_elapsed_time.count()) / 1000 << "ms" << std::endl;

  return 0;
}