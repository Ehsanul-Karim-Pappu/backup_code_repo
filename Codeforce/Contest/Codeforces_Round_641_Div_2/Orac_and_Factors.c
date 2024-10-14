#include <stdio.h>
#include <math.h>

int prime(int n)
{
    int i, flag = 1;
    for (i = 2; i <= sqrt(n); i++)
    {
        if (n % i == 0)
        {
            flag = 0;
            break;
        }
    }

    if (flag == 1)
        return 1;
    else
        return i;
}

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
    int test, n, k, i;

    scanf("%d", &test);

    while (test--)
    {
        scanf("%d %d", &n, &k);

        while (k--)
        {
            if (prime(n) == 1)
            {
                n += n;
                continue;
            }
            else if (n % 2 == 0)
            {
                n += ((k + 1) * 2);
                break;
            }
            else
            {
                i = prime(n);
                n += i;
                continue;
            }
        }
        printf("%d\n", n);
    }

    return 0;
}