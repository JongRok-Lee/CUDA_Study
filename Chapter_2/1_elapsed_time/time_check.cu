#include <iostream>
#include <chrono>

#define MAX 10000

void CPUBigJob()
{
  int cnt = 0;
  for (int i = 0; i < MAX; i++) {
    for (int j = 0; j < MAX; j++) {
      cnt++;
    }
  }
}

int main()
{
  // CPU Time Check
  std::chrono::steady_clock::time_point cpu_start = std::chrono::steady_clock::now();
  CPUBigJob();
  std::chrono::steady_clock::time_point cpu_end = std::chrono::steady_clock::now();
  std::chrono::milliseconds elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
  std::cout << "CPU Elapsed time: " << elapsed_time.count() << "ms" << std::endl;

  // Wall-Clock Time Check
  std::chrono::system_clock::time_point wall_start = std::chrono::system_clock::now();
  CPUBigJob();
  std::chrono::system_clock::time_point wall_end = std::chrono::system_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start);
  std::cout << "Wall-Clock Elapsed time: " << elapsed_time.count() << "ms" << std::endl;

  return 0;

}