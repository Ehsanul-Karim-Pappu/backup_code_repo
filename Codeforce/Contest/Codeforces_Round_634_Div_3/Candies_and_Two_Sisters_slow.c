#include <stdio.h>

int main()
{
    int t, a, b, n, i, count;

    scanf("%d", &t);

    for (i = 0; i < t; i++)
    {
        count = 0;
        scanf("%d", &n);

        for (b = 1; b <= n / 2; b++)
        {
            a = n - b;
            if (a > b)
                count++;
        }
        printf("%d\n", count);
    }
}