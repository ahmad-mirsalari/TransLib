#include <stdio.h>
#include <math.h>

#define ECG_LEN 9
#define NC 4

int main()
{
  int input_data[ECG_LEN]={2,4,6,1,3,4,5,6};
  int h[NC]={2,1,4,1};
  int y[ECG_LEN]={0};

  int n=0;
  int k=0;

  int N = ECG_LEN + NC;
  int delta = floor(NC/2);
  printf("delta %d\n", delta);

  for (n = 1+delta; n <= N-delta; n++)
  {
    y[n]=0;
    for (k = 0; k <= ECG_LEN; k++)
    {
        if(n - k + 1 >= 1 && n - k + 1 <= NC)
        y[n-delta] += input_data[k] * h[n-k+1];
    }
    printf("y[%d]:%d\n", n-delta, y[n-delta]);
  }
  return 0;
}
