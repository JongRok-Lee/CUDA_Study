#include <iostream>
#include <chrono>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void setRandomData(float data[], const int size, const float bound=static_cast<float>(1000)) {
  int bnd = static_cast<int>(bound);
  for (int i = 0; i < size; i++) {
    data[i] = (rand() % bnd) / bound;
  }
}

__global__ void matEleMulGPU(float matC[], const float matA[], const float matB[],
               const uint32_t nz, const uint32_t ny, const uint32_t nx) {
  uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < nx && y < ny && z < nz) {
    uint32_t idx = (z * ny + y) * nx + x;
    matC[idx] = matA[idx] * matB[idx];
  }
}

__host__ __device__ inline uint32_t ceilDiv(const uint32_t a, const uint32_t b) {
  return (a + b - 1) / b;
}


namespace chrono = std::chrono;

int main()
{
  const dim3 dimImage(300, 300, 256);
  const uint32_t SIZE = dimImage.x * dimImage.y * dimImage.z;

  float *A = new float[SIZE];
  float *B = new float[SIZE];
  float *C = new float[SIZE];
  srand(0);
  setRandomData(A, SIZE);
  setRandomData(B, SIZE);


  float *d_A, *d_B, *d_C;
  float size = SIZE * sizeof(float);
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  dim3 blockDim(8, 8, 8);
  dim3 gridDim(ceilDiv(dimImage.x, blockDim.x), ceilDiv(dimImage.y, blockDim.y), ceilDiv(dimImage.z, blockDim.z));
  std::cout << "Grid: " << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << std::endl;
  std::cout << "Block: " << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << std::endl;

  chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  matEleMulGPU<<<gridDim, blockDim>>>(d_C, d_A, d_B, dimImage.z, dimImage.y, dimImage.x);
  cudaDeviceSynchronize();
  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  chrono::microseconds time = chrono::duration_cast<chrono::microseconds>(end - begin);
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  std::cout << "Time: " << time.count() << " microseconds" << std::endl;
  std::cout.precision(10);
  std::cout << "VecA:\t" << A[0] << ", " << A[1] << ", " << A[2] << ", " << A[3] << std::endl;
  std::cout << "VecB:\t" << B[0] << ", " << B[1] << ", " << B[2] << ", " << B[3] << std::endl;
  std::cout << "VecC:\t" << C[0] << ", " << C[1] << ", " << C[2] << ", " << C[3] << std::endl;

  delete[] A, B, C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}