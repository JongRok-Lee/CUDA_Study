#include <iostream>
#include <chrono>
#include <cstdlib>

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
__global__ void kernelVecAddGPU(PREC vecC[], const PREC vecA[], const PREC vecB[], const int size) {
  for (int i = 0; i < size; i++) {
    vecC[i] = vecA[i] + vecB[i];
  }
}

int main()
{
  const int SIZE = 1024 * 1024;
  float *vecA = new float[SIZE];
  float *vecB = new float[SIZE];
  float *vecC = new float[SIZE];

  srand(0);
  setRandomData(vecA, SIZE);
  setRandomData(vecB, SIZE);

  float *d_vecA, *d_vecB, *d_vecC;
  cudaMalloc(&d_vecA, SIZE * sizeof(float));
  cudaMalloc(&d_vecB, SIZE * sizeof(float));
  cudaMalloc(&d_vecC, SIZE * sizeof(float));
  cudaMemcpy(d_vecA, vecA, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vecB, vecB, SIZE * sizeof(float), cudaMemcpyHostToDevice);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  kernelVecAddGPU<<<1, 1>>>(d_vecC, d_vecA, d_vecB, SIZE);
  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::microseconds elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  std::cout << "GPU Elapsed time: " << static_cast<float>(elapsed_time.count()) / 1000 << "ms" << std::endl;

  cudaMemcpy(vecC, d_vecC, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  float sumA = getSum(vecA, SIZE);
  float sumB = getSum(vecB, SIZE);
  float sumC = getSum(vecC, SIZE);
  float sumDiff = fabsf(sumC - (sumA + sumB));

  std::cout << "Sum A: " << sumA << std::endl;
  std::cout << "Sum B: " << sumB << std::endl;
  std::cout << "Sum C: " << sumC << std::endl;
  std::cout << "Sum Difference: " << sumDiff << std::endl;

  delete[] vecA;
  delete[] vecB;
  delete[] vecC;
  cudaFree(d_vecA);
  cudaFree(d_vecB);
  cudaFree(d_vecC);

  return 0;
}