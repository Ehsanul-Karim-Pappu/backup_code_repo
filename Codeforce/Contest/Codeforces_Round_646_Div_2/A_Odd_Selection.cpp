#include <bits/stdc++.h>

using namespace std;

int main()
{
    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, x;
        scanf("%d %d", &n, &x);

        int t, e = 0, o = 0;
        for (int i = 0; i < n; i++)
        {
            scanf("%d", &t);
            if (t % 2)
                o++;
        }
        e = n - o;
        if ((o >= x && (x % 2 || (!(x % 2) && e >= 1))) ||
            (o && o < x && (o % 2 or (!(o % 2) && o - 1 + e >= x))))
            printf("Yes\n");
        else
            printf("No\n");
    }

    return 0;
}