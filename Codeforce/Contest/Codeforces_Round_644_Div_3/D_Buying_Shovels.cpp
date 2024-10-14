#include <bits/stdc++.h>

using namespace std;

#define ll long long int

ll factors[200000];

ll factor(ll number)
{
    ll j = 0;

    for (ll i = 2; i * i <= number; i++)
    {
        if (number % i == 0 && i != number)
        {
            factors[j++] = i;
            if (number / i != i)
                factors[j++] = number / i;
        }
    }
    return j;
}

bool isPrime(ll n)
{
    if (n < 2)
        return false;

    for (ll i = 2; i * i <= n; i++)
    {
        if (n % i == 0)
            return false;
    }
    return true;
}

int main()
{
    freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    ll test;
    scanf("%lld", &test);

    while (test--)
    {
        ll n, k, size, i;
        bool flag = true;
        scanf("%lld %lld", &n, &k);

        if (n <= k)
            printf("1\n");
        else if (isPrime(n) || k == 1)
            printf("%lld\n", n);
        else
        {
            size = factor(n);
            sort(factors, factors + size);
            for (i = size - 1; i >= 0; i--)
            {
                if (factors[i] <= k)
                {
                    flag = false;
                    break;
                }
            }
            if (!flag)
                printf("%lld\n", n / factors[i]);
            else
                printf("%lld\n", n);
        }
    }
    return 0;
}