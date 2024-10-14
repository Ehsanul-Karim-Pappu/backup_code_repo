#include <stdio.h>

int main()
{
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    int test, n, i, j, k, a[10000], r[10000], m;

    scanf("%d", &test);

    while (test--)
    {
        m = 1;
        k = 0;
        scanf("%d", &n);
        for (i = 0; n != 0; i++)
        {
            r[i] = n % 10; //putting the digits of the number in an array
            n = n / 10;
        }
        for (j = 0; j < i; j++)
        {
            if (j != 0)
                m *= 10;
            if (r[j] != 0)
            {
                a[k] = r[j] * m;
                k++;
            }
        }

        printf("%d\n", k);
        for (i = 0; i < k; i++)
            printf("%d ", a[i]);

        printf("\n");
    }

    return 0;
}