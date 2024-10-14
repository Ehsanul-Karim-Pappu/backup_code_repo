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

        int a[1001], e = 0, o = 0, mx = 0, cont[1001] = {0};
        for (int i = 0; i < n; i++)
        {
            scanf("%d", &a[i]);
            mx = max(mx, a[i]);
            if (a[i] % 2 == 0)
                e++;
            else
                o++;
            cont[a[i]]++;
        }

        for (int i = 1; i <= mx; i += 2)
        {
            o += cont[i];
            
        }

        if (x % 2 == 1 && o >= x)
            printf("Yes\n");
        else if (x % 2 == 1 && o < x && o % 2 == 1)
            printf("Yes\n");
        else if (x % 2 == 1 && o < x && o % 2 == 0)
            printf("No\n");
        /*else if (x % 2 == 0 && o > x)
            printf("Yes\n");
        else if (x % 2 == 0 && o == x)
            printf("No\n");
        else if (x % 2 == 0 && o < x && o % 2 == 1)
            printf("Yes\n");
        else if (x % 2 == 0 && o < x && o % 2 == 0)
            printf("No\n");
        else
            printf("No\n");*/
    }

    return 0;
}