#include <bits/stdc++.h>

using namespace std;

int a[10];
int main()
{
    freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, mx = 0, t, ans = 0;
        bool isEq = true, lessPaise = false;
        scanf("%d", &n);
        for (int i = 0; i < n; i++)
        {
            scanf("%d", &a[i]);
            if (!i)
                t = a[i];

            if (t > a[i])
            {
                lessPaise = true;
                // ans = i;
            }

            if (a[i] != t)
            {
                isEq = false;
                t = a[i];
            }

            if (mx <= a[i])
            {
                if (lessPaise && mx == a[i])
                {
                    ans = i;
                    lessPaise = false;
                }
                mx = a[i];
                if (i == n - 1 && !lessPaise)
                    ans = i;
            }
        }
        if (isEq)
            printf("-1\n");
        else
            printf("%d\n", ans + 1);
    }

    return 0;
}