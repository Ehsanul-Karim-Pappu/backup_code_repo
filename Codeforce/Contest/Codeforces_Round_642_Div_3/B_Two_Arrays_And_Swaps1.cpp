#include <bits/stdc++.h>

using namespace std;

void swap(int *a, int *b)
{
    int t;

    t = *b;
    *b = *a;
    *a = t;
}

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    int t, n, k, i, x;
    scanf("%d", &t);
    while (t--)
    {
        int sum = 0;
        scanf("%d %d", &n, &k);
        int a[n], b[n];

        for (i = 0; i < n; i++)
            scanf("%d", &a[i]);
        for (i = 0; i < n; i++)
            scanf("%d", &b[i]);

        sort(a, a + n);
        sort(b, b + n, greater<int>());

        for (i = 0; i < k; i++)
        {
            if (a[i] < b[i])
                swap(&a[i], &b[i]);
        }

        for (i = 0; i < n; i++)
            sum += a[i];

        printf("%d\n", sum);
    }
    return 0;
}
