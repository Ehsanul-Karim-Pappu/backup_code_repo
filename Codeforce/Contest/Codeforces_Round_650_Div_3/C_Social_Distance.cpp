#include <bits/stdc++.h>

using namespace std;

int main()
{
    //freopen("input.txt", "r", stdin);

    int t;
    scanf("%d", &t);

    while (t--)
    {
        int n, k;
        scanf("%d %d", &n, &k);
        getchar();

        char str[20];
        scanf("%s", str);

        int first = -1, last = -2;
        for (int i = 0; i < n; i++)
        {
            if (str[i] == '1')
            {
                if (first == -1)
                    first = i;
                last = i;
            }
        }
        if (first == -1)
        {
            printf("%d\n", (int)ceil(n * 1. / (k + 1)));
            continue;
        }
        int c = 0;
        if (first > k)
        {
            str[0] = '1';
            c++;
            first = 0;
        }
        if (n - last - 2 >= k)
        {
            str[n - 1] = '1';
            c++;
            last = n - 1;
        }

        bool flag = false;
        int i, j;
        for (i = first; i < n;)
        {
            for (j = first + 1; j <= first + k * 2 + 1 && j < n; j++)
            {
                if (str[j] == '1')
                {
                    flag = true;
                    i = first = j;
                    break;
                }
            }
            if (!flag && first != last)
            {
                i = first = first + k + 1;
                str[first] = '1';
                c++;
            }
            else if (first == last)
                break;
            flag = false;
        }
        printf("%d\n", c);
    }

    return 0;
}