#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);

    int a[100000], test;
    scanf("%d", &test);

    while (test--)
    {
        int n, x;
        scanf("%d %d", &n, &x);

        int mn = INT_MAX, sum = 0;
        bool flag = true;
        for (int i = 0; i < n; i++)
        {
            scanf("%d", &a[i]);
            if (a[i] % x)
            {
                flag = false;
                printf("%d\n", max(i + 1, n - i));
                break;
            }
        }
        if(flag)
            printf("-1\n");
        else
            continue;
    }

    return 0;
}