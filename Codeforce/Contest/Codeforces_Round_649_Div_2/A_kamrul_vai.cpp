#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);

    int a[100000], cul[100000], test;
    scanf("%d", &test);

    while (test--)
    {
        int n, x;
        scanf("%d%d", &n, &x);

        int mn = INT_MAX, sum = 0;
        for (int i = 0; i < n; i++)
        {
            scanf("%d", &a[i]);
            sum += a[i];
            cul[i] = sum;
            if (sum % x != 0)
                mn = min(mn, i);
        }

        int mx = INT_MIN;
        for (int i = 0; i < n; i++)
        {
            if (cul[i] % x == 0)
                mx = max(mx, i - mn);
            else
                mx = i + 1;
        }

        if (mn == INT_MAX)
            printf("-1\n");
        else
            printf("%d\n", mx);
    }

    return 0;
}