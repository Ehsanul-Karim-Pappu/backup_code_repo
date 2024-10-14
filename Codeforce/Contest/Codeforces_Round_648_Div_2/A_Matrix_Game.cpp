#include <bits/stdc++.h>

using namespace std;

int main()
{
    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, m;
        scanf("%d %d", &n, &m);

        int a[55][55];
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
                scanf("%d", &a[i][j]);
        }

        bool flag = false, As = true, Vi = false;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                if (!a[i][j] && As)
                {
                    if (a[i][j - 1])
                    {
                        flag = true;
                        continue;
                    }
                    else if (a[i][j + 1])
                    {
                        flag = true;
                        continue;
                    }
                    else if (a[i - 1][j])
                    {
                        flag = true;
                        continue;
                    }
                    else if (a[i + 1][j])
                    {
                        flag = true;
                        continue;
                    }
                    else
                    {
                        flag = false;
                        a[i][j] = 1;
                    }
                    Vi = true;
                    As = false;
                }
                else if (!a[i][j] && Vi)
                {
                    if (a[i][j - 1])
                    {
                        flag = true;
                        continue;
                    }
                    else if (a[i][j + 1])
                    {
                        flag = true;
                        continue;
                    }
                    else if (a[i - 1][j])
                    {
                        flag = true;
                        continue;
                    }
                    else if (a[i + 1][j])
                    {
                        flag = true;
                        continue;
                    }
                    else
                    {
                        flag = false;
                        a[i][j] = 1;
                    }
                    As = true;
                    Vi = false;
                }
            }
        }
    }

    return 0;
}