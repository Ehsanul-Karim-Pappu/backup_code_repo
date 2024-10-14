#include <bits/stdc++.h>

using namespace std;

int main()
{
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, a[2000], max_num = -1, ans = 0;
        bool max_num_index_odd = false;

        scanf("%d", &n);
        for (int i = 1; i <= n; i++)
        {
            scanf("%d", &a[i]);

            if (a[i] >= max_num)
                max_num = a[i];
        }

        for (int i = 1; i <= n; i++)
        {
            if (a[i] == max_num)
            {
                if (i % 2 != 0)
                {
                    max_num_index_odd = true;
                    break;
                }
            }
        }

        if (n % 2 != 0 && max_num_index_odd) // if size is odd
            ans = max_num + (n / 2) + 1;
        else
            ans = max_num + n / 2;

        printf("%d\n", ans);
    }

    return 0;
}