#include <bits/stdc++.h>

using namespace std;

int a[10000];
int main()
{
    //freopen("input.txt", "r", stdin);

    int t;
    scanf("%d", &t);

    while (t--)
    {
        int n, c = 0;
        scanf("%d", &n);

        for (int i = 0; i < n; i++)
            scanf("%d", &a[i]);

        for (int i = 0; i < n; i++)
        {
            if (i % 2 != a[i] % 2)
            {
                if (a[i] % 2 == 1)
                {
                    for (int j = i + 1; j < n; j += 2)
                    {
                        if (a[j] % 2 == 0)
                        {
                            swap(a[i], a[j]);
                            c++;
                            break;
                        }
                    }
                }
                else
                {
                    for (int j = i + 1; j < n; j += 2)
                    {
                        if (a[j] % 2 == 1)
                        {
                            swap(a[i], a[j]);
                            c++;
                            break;
                        }
                    }
                }
            }
        }
        bool flag = false;
        for (int i = 0; i < n; i++)
        {
            if (i % 2 != a[i] % 2)
                flag = true;
        }

        if (flag)
            printf("-1\n");
        else
            printf("%d\n", c);
    }

    return 0;
}