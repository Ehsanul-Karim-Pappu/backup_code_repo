#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);

    int t;
    scanf("%d", &t);

    while (t--)
    {
        int a, b;
        scanf("%d %d", &a, &b);

        int mn = min(a, b), mx = max(a, b);

        if (mn * 2 <= mx)
        {
            printf("%d\n", mn);
            continue;
        }
        else
        {
            int r, q, t, diff, ans;
            r = mn % 3;
            q = mn / 3;
            t = q * 3;
            diff = mx - t;
            if (r == 2 && diff >= 4)
            {
                ans = q * 2 + 2;
                printf("%d\n", ans);
                continue;
            }
            else if (r == 2 && diff >= 2)
            {
                ans = q * 2 + 1;
                printf("%d\n", ans);
                continue;
            }
            else if (r == 1 && diff >= 2)
            {
                ans = q * 2 + 1;
                printf("%d\n", ans);
                continue;
            }
            else
            {
                ans = q * 2;
                printf("%d\n", ans);
                continue;
            }
        }
    }

    return 0;
}