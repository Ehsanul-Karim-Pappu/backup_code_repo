#include <bits/stdc++.h>

using namespace std;

int e[200000];

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
    int test, n;

    scanf("%d", &test);

    while (test--)
    {
        scanf("%d", &n);
        for (int i = 0; i < n; i++)
            scanf("%d", &e[i]);

        sort(e, e + n);

        int ans = 0, ref = 0;
        for (int i = 0; i < n; i++)
        {
            ref++;
            if (ref == e[i])
            {
                ans++;
                ref = 0;
            }
        }
        printf("%d\n", ans);
    }

    return 0;
}