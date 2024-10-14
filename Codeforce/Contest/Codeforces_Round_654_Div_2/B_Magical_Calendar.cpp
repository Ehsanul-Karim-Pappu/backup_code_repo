#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        long long n, r, ans = 0;
        scanf("%lld %lld", &n, &r);

        if (n <= r)
        {
            n--;
            ans = (n * (n + 1)) / 2;
            ans++;
        }
        else
            ans = (r * (r + 1)) / 2;

        printf("%lld\n", ans);
    }

    return 0;
}