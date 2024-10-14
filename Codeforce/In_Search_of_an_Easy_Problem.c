#include<stdio.h>
int main()
{
    int n[100],element,sum=0;
    scanf("%d",&element);

    for(int i=0; i<element; ++i)
    {
        scanf("%d", &n[i]);
        sum+= n[i];
    }
    if (sum==0)
        printf("EASY");
    else
        printf("HARD");

}
