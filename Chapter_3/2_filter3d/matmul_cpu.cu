#include <iostream>
#include <chrono>
#include <cstdlib>

#include <cuda.h>

void setRandomData(float data[], const int size, const float bound=static_cast<float>(1000)) {
  int bnd = static_cast<int>(bound);
  for (int i = 0; i < size; i++) {
    data[i] = (rand() % bnd) / bound;
  }
}

void matEleMulCPU(float matC[], const float matA[], const float matB[],
               const uint32_t nz, const uint32_t ny, const uint32_t nx) {
  for (uint32_t z = 0; z < nz; z++) {
    for (uint32_t y = 0; y < ny; y++) {
      for (uint32_t x = 0; x < nx; x++) {
        uint32_t idx = (z * ny + y) * nx + x;
        matC[idx] = matA[idx] * matB[idx];
      }
    }
  }
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

  chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  matEleMulCPU(C, A, B, dimImage.z, dimImage.y, dimImage.x);
  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  chrono::microseconds time = chrono::duration_cast<chrono::microseconds>(end - begin);

  std::cout << "Time: " << time.count() << " microseconds" << std::endl;
  std::cout.precision(10);
  std::cout << "VecA:\t" << A[0] << ", " << A[1] << ", " << A[2] << ", " << A[3] << std::endl;
  std::cout << "VecB:\t" << B[0] << ", " << B[1] << ", " << B[2] << ", " << B[3] << std::endl;
  std::cout << "VecC:\t" << C[0] << ", " << C[1] << ", " << C[2] << ", " << C[3] << std::endl;

  delete[] A, B, C;
  return 0;
}