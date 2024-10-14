#include <stdio.h>
#include <math.h>

int main()
{
    int test, n, x, j, sum, term;

    scanf("%d", &test);

    while (test--)
    {
        scanf("%d", &n);
        for (x = 1; sum != n; x++)
        {
            sum = 0;
            for (j = 0; sum < n; j++)
            {
                term = pow(2, j);
                sum += (term * x);
            }
            if (sum == n)
            {
                printf("%d\n", x);
            }
            printf("y");
        }
    }

    return 0;
}