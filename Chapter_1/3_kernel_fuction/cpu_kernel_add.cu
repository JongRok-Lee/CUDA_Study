#include <stdio.h>

void add_kernel(int idx, const int *a, const int *b, int *c)
{
  c[idx] = a[idx] + b[idx];
}

int main()
{
  const int SIZE = 5;
  const int a[SIZE] = {1, 2, 3, 4, 5};
  const int b[SIZE] = {10, 20, 30, 40, 50};
  int c[SIZE] = {0,};

  for (int i = 0; i < SIZE; i++) {
    add_kernel(i, a, b, c);
  }

  printf("c[i] = a[i] + b[i]\n");
  printf("%d %d %d %d %d\n", c[0], c[1], c[2], c[3], c[4]);

  return 0;
}