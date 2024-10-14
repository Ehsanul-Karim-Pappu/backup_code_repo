#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;
        scanf("%d", &n);
        int x = 1, y = n - 1, mn = INT_MAX;
        while (x <= y)
        {
            //printf("lcm(%d, %d) = %d\n", x, y, x * y / __gcd(x, y));
            mn = min(mn, x * y / __gcd(x, y));
            x++, y--;
        }
        printf("%d %d\n", min(mn, n - mn), max(mn, n - mn));
    }

    return 0;
}