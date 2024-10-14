#include <stdio.h>

long long digit[100];

long long digit_sep(long long number)
{
    long long i;
    for (i = 0; number != 0; i++)
    {
        digit[i] = number % 10;
        number /= 10;
    }
    return i;
}

long long check_zero(long long size)
{
    long long i;

    for (i = 0; i < size; i++)
    {
        if (digit[i] == 0)
            return 1;
    }
    return 0;
}

long long max(long long *array, long long size)
{
    long long i, mx;
    mx = array[0];

    for (i = 0; i < size; i++)
    {
        if (array[i] > mx)
            mx = array[i];
    }
    return mx;
}

long long min(long long *array, long long size)
{
    long long i, mn;
    mn = array[0];

    for (i = 0; i < size; i++)
    {
        if (array[i] < mn)
            mn = array[i];
    }
    return mn;
}

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    long long test, a1, k, i, sum, size;

    scanf("%lld", &test);

    while (test--)
    {
        sum = 0;
        scanf("%lld %lld", &a1, &k);

        if (k == 1)
        {
            printf("%lld\n", a1);
            continue;
        }
        else
        {
            sum = a1;
            for (i = 1; i < k; i++)
            {
                size = digit_sep(sum);
                if (check_zero(size))
                    break;
                else
                    sum += (min(digit, size) * max(digit, size));
            }
        }
        printf("%lld\n", sum);
    }

    return 0;
}