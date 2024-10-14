#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, m, x, y, c = 0, o = 0, e = 0, total = 0;

        scanf("%d %d %d %d", &n, &m, &x, &y);
        char ch[n][m];
        getchar();

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                scanf("%c", &ch[i][j]);
                if (ch[i][j] == '.')
                {
                    c++;
                    total++;
                }

                if ((ch[i][j] == '.' && j == m - 1) || ch[i][j] == '*')
                {
                    if (c % 2 == 0)
                    {
                        e += c / 2;
                        c = 0;
                    }
                    else
                    {
                        e += c / 2;
                        o++;
                        c = 0;
                    }
                }
            }
            getchar();
        }

        if (x * 2 <= y)
            printf("%d\n", x * total);
        else
            printf("%d\n", e * y + o * x);
    }
    return 0;
}