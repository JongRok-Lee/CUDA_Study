#include <iostream>
#include <chrono>

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

int main()
{
  const unsigned int SIZE = 256 * 1024 * 1024;
  float saxpy_a = 1.234f;
  float *vecX = new float[SIZE];
  float *vecY = new float[SIZE];
  float *vecZ = new float[SIZE];

  srand(0);
  setRandomData(vecX, SIZE);
  setRandomData(vecY, SIZE);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (unsigned int i = 0; i < SIZE; i++) {
    vecZ[i] = saxpy_a * vecX[i] + vecY[i];
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::microseconds time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  std::cout << "Time: " << time.count() << " us" << std::endl;

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

  delete[] vecX, vecY, vecZ;
}