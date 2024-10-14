#include <bits/stdc++.h>

using namespace std;

int main()
{
    //freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int a, b, n;
        scanf("%d %d %d", &a, &b, &n);

        bool flag = true;
        int c = 0;
        if (a >= b)
        {
            while (a <= n && b <= n)
            {
                c++;
                if (flag)
                {
                    b += a;
                    flag = false;
                }
                else
                {
                    a += b;
                    flag = true;
                }
            }
        }
        else
        {
            while (a <= n && b <= n)
            {
                c++;
                if (flag)
                {
                    a += b;
                    flag = false;
                }
                else
                {
                    b += a;
                    flag = true;
                }
            }
        }
        printf("%d\n", c);
    }
    return 0;
}