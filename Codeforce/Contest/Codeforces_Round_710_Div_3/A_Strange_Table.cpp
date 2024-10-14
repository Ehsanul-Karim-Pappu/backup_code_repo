#include <bits/stdc++.h>

using namespace std;

int main()
{
    // freopen("input.txt","r",stdin);

    long long int test;
    scanf("%lld", &test);

    while (test--)
    {
        long long int x, n, m, row, col, ans;
        scanf("%lld %lld %lld", &n, &m, &x);

        col = (int)ceil((double)x / (double)n);
        row = x % n;
        if (!row) row = n;

        ans = (row - 1) * m + col;

        printf("%lld\n", ans);
    }

    return 0;
}