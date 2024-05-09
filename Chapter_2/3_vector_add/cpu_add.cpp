#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

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
void kernelVecAddCPU(PREC vecC[], const PREC vecA[], const PREC vecB[], const int size) {
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

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  kernelVecAddCPU(vecC, vecA, vecB, SIZE);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::microseconds elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  std::cout << "CPU Elapsed time: " << static_cast<float>(elapsed_time.count()) / 1000 << "ms" << std::endl;

  float sumA = getSum(vecA, SIZE);
  float sumB = getSum(vecB, SIZE);
  float sumC = getSum(vecC, SIZE);
  float sumDiff = fabsf(sumA + sumB - sumC);

  std::cout << "Sum A: " << sumA << std::endl;
  std::cout << "Sum B: " << sumB << std::endl;
  std::cout << "Sum C: " << sumC << std::endl;
  std::cout << "Sum Difference: " << sumDiff << std::endl;

  delete[] vecA;
  delete[] vecB;
  delete[] vecC;

  return 0;
}