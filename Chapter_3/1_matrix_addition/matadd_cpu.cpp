#include <iostream>
#include <chrono>
#include <cstdlib>

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

void matAddCPU(float matC[], const float matA[], const float matB[],
               const uint32_t nrow, const uint32_t ncol) {
  for (uint32_t row = 0; row < nrow; row++) {
    for (uint32_t col = 0; col < ncol; col++) {
      uint32_t idx = row * ncol + col;
      matC[idx] = matA[idx] + matB[idx];
    }
  }
}


namespace chrono = std::chrono;

int main()
{
  const uint32_t nrow = 10000;
  const uint32_t ncol = 10000;

  float *A = new float[nrow * ncol];
  float *B = new float[nrow * ncol];
  float *C = new float[nrow * ncol];
  srand(0);
  setRandomData(A, nrow * ncol);
  setRandomData(B, nrow * ncol);

  chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  matAddCPU(C, A, B, nrow, ncol);
  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  chrono::microseconds time = chrono::duration_cast<chrono::microseconds>(end - begin);

  float sumA = getSum(A, nrow * ncol);
  float sumB = getSum(B, nrow * ncol);
  float sumC = getSum(C, nrow * ncol);
  std::cout << "Time: " << time.count() << " microseconds" << std::endl;
  std::cout.precision(10);
  std::cout << "Sum of A: " << sumA << std::endl;
  std::cout << "Sum of B: " << sumB << std::endl;
  std::cout << "Sum of C: " << sumC << std::endl;
  std::cout << "Diff: " << sumC - (sumA + sumB) << std::endl;
  std::cout << "VecA:\t" << A[0] << ", " << A[1] << ", " << A[2] << ", " << A[3] << std::endl;
  std::cout << "VecB:\t" << B[0] << ", " << B[1] << ", " << B[2] << ", " << B[3] << std::endl;
  std::cout << "VecC:\t" << C[0] << ", " << C[1] << ", " << C[2] << ", " << C[3] << std::endl;

  delete[] A, B, C;
  return 0;
}