#include<stdio.h>

int main()
{
  int n,k;
  scanf("%d",&n);
  scanf("%d",&k);
  for(k;k>0;k--)
  {
      if (n%10==0){
        n=n/10;
      }
      else
        n=n-1;
      printf("k= %d\n",k);
      printf("n= %d\n",n);

  }
  //printf("%d",n);

}
