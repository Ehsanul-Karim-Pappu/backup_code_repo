#include <bits/stdc++.h>

using namespace std;

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;

        scanf("%d", &n);
        getchar();
        char matrix[n][n + 1];

        for (int i = 0, j = 0; i <= n && j < n; i++)
        {
            scanf("%c", &matrix[j][i]);
            if (matrix[j][i] == '\n')
            {
                matrix[j][i] = '\0';
                i = -1;
                j++;
            }
        }

        bool flag = false;

        for (int i = 0, j = 0; i < n && j < n; i++)
        {
            if (matrix[j][i] == '1' && i != n - 1 && matrix[j][i + 1] == '0' && matrix[j + 1][i] == '0')
            {
                flag = true;
                break;
            }
            if (i == n - 1)
            {
                i = -1;
                j++;
            }
        }

        if (flag)
            printf("NO\n");
        else
            printf("YES\n");
    }

    return 0;
}