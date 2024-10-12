#include <bits/stdc++.h>

using namespace std;

int main()
{
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, k, ans = 0;
        scanf("%d", &n);
        scanf("%d", &k);

        if (k == 1)
            printf("%d\n", n);
        else if (n < k)
            printf("%d\n", n);
        else if (n == k)
            printf("1\n");
        else
        {
            while (n >= k)
            {
                ans++;
                long long power = 1;
                while (power * k <= n)
                {
                    power *= k;
                }
                n -= power;
                // printf("n = %d\n", n);
            }

            printf("%d\n", ans + n);
        }
    }

    return 0;
}