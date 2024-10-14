#include <stdio.h>
#include <math.h>

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
    int test, n, x, i, term, sum;
    scanf("%d", &test);

    while (test--)
    {
        scanf("%d", &n);
        sum = 0;

        for (i = 0; i < 30; i++)
        {
            term = pow(2, i);
            sum += term;

            if (n % sum == 0 && sum != 1)
            {
                x = n / sum;
                printf("%d\n", x);
                break;
            }
        }
    }

    return 0;
}