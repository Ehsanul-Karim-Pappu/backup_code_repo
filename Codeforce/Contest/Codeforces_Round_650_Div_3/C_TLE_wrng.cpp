#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);

    int t;
    scanf("%d", &t);

    while (t--)
    {
        int n, k, i, j;
        scanf("%d %d", &n, &k);
        getchar();

        char str[20];
        scanf("%s", str);

        int index = -1;
        for (i = 0; i < n; i++)
        {
            if (str[i] == '1')
            {
                index = i;
                break;
            }
        }
        if (index == -1)
        {
            printf("%d\n", ceil(n * 1. / (k - 1)));
            continue;
        }

        bool flag = false;
        int c = 0;
        for (i = index; i < n; i++)
        {
            if (str[i] == '0')
            {
                for (j = i + 1; j < n && j <= i + k; j++)
                {
                    if (str[j] == '1')
                    {
                        flag = true;
                        break;
                    }
                }
                if (flag)
                {
                    flag = false;
                    continue;
                }
                for (j = i - 1; j >= 0 && j >= i - k; j--)
                {
                    if (str[j] == '1')
                    {
                        flag = true;
                        break;
                    }
                }
                if (!flag)
                {
                    c++;
                    str[i] = '1';
                }
                flag = false;
            }
        }
        printf("%d\n", c);
    }

    return 0;
}