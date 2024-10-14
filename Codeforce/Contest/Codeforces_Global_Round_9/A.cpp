#include <bits/stdc++.h>

using namespace std;

int a[100000];
int main()
{
    freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;
        scanf("%d", &n);
        for (int i = 0; i < n; i++)
            scanf("%d", &a[i]);

        for (int i = 1; i <= n / 2; i++)
        {
            if (a[i] < a[i - 1])
                a[i - 1] = -a[i - 1];
        }
        for (int i = (n / 2) + 1; i < n; i++)
        {
            if (a[i] > a[i - 1])
                a[i] = -a[i];
        }
        for (int i = 0; i < n; i++)
            printf("%d ", a[i]);
        printf("\n");
    }

    return 0;
}