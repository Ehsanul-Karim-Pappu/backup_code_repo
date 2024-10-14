#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, s[1024];
        scanf("%d", &n);

        for (int i = 0; i < n; i++)
            scanf("%d", &s[i]);
        sort(s, s + n);

        int ans[1024], i;
        bool flag = true;
        for (i = 1; i <= 1024; i++)
        {
            if (flag)
            {
                for (int j = 0; j < n; j++)
                    ans[j] = i ^ s[j];
                sort(ans, ans + n);

                for (int i = 0; i < n; i++)
                {
                    if (ans[i] != s[i])
                    {
                        flag = true;
                        break;
                    }
                    else
                        flag = false;
                }
            }
            else
                break;
        }

        if (!flag)
            printf("%d\n", i - 1);
        else
            printf("-1\n");
    }
    return 0;
}