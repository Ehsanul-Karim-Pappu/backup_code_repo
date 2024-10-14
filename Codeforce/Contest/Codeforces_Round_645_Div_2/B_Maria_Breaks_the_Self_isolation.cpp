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
        int n, a[200000];
        scanf("%d", &n);

        for (int i = 0; i < n; i++)
            scanf("%d", &a[i]);

        sort(a, a + n);

        int i, granny = 0;
        for (i = 0; i < n; i++)
        {
            if (a[i] <= i + 1)
            {
                granny = i + 1;
                continue;
            }
        }
        printf("%d\n", granny + 1);
    }

    return 0;
}