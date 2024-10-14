#include <bits/stdc++.h>

using namespace std;

int main()
{
    //freopen("input.txt", "r", stdin);
    int test;
    scanf("%d", &test);

    while (test--)
    {
        long long x, y, n;
        scanf("%lld %lld %lld", &x, &y, &n);
        if (n % x != y && n % x < y)
            n = n - (n % x) - (x - y);
        if (n % x != y && n % x > y)
            n = n - ((n % x) - y);
        if (n < 0)
            n = 0;
        printf("%lld\n", n);
    }

    return 0;
}