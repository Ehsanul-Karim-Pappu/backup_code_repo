#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);

    int t, n, i, j, x, a[100000];
    scanf("%d", &t);

    while (t--)
    {
        int sum = 0, mx = 0, ans = 0;
        bool flag = true;
        scanf("%d %d", &n, &x);
        for (i = 0; i < n; i++)
        {
            scanf("%d", &a[i]);
            sum += a[i];
        }
        int s = sum;
        if (sum % x)
        {
            printf("%d\n", n);
            continue;
        }
        else
        {
            for (i = 0; i < n; i++)
            {
                sum = s;
                sum -= a[i];
                if (sum % x)
                {
                    ans = n - i - 1;
                    flag = false;
                    break;
                }
            }
            mx = max(ans, mx);
            sum = s;
            for (i = n - 1; i > mx; i--)
            {
                sum -= a[i];
                if (sum % x)
                {
                    ans = i;
                    flag = false;
                    break;
                }
            }
            mx = max(ans, mx);

            if (flag)
                printf("-1\n");
            else
                printf("%d\n", mx);
        }
    }
}
