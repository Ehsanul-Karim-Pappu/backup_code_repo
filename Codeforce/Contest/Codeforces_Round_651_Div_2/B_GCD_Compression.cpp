#include <bits/stdc++.h>

using namespace std;

int a[10000];
int main()
{
    //freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;
        scanf("%d", &n);

        int odd[10000], even[10000], o = 0, e = 0;
        for (int i = 1, j = 1, k = 1; i <= 2 * n; i++)
        {
            scanf("%d", &a[i]);
            if (a[i] % 2)
            {
                odd[j++] = i;
                o++;
            }
            else
            {
                even[k++] = i;
                e++;
            }
        }

        if (o % 2)
            o--, e--;
        else if (e == 0)
            o -= 2;
        else if (o == 0)
            e -= 2;
        else
            o -= 2;

        for (int i = 1, j = 1; i <= o / 2; i++, j += 2)
            printf("%d %d\n", odd[j], odd[j + 1]);
        for (int i = 1, j = 1; i <= e / 2; i++, j += 2)
            printf("%d %d\n", even[j], even[j + 1]);
    }

    return 0;
}