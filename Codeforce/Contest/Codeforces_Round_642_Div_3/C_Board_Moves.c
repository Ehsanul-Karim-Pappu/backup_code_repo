#include <stdio.h>

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    long long t, n, i, ans, m;
    scanf("%lld", &t);

    while (t--)
    {
        m = 0;
        ans = 0;
        scanf("%lld", &n);

        if (n == 1)
            printf("0\n");
        else
        {
            for (i = 1; i <= n / 2; i++)
            {
                m += 8;
                ans += (m * i);
            }
            printf("%lld\n", ans);
        }
    }
    return 0;
}