#include <bits/stdc++.h>

using namespace std;

int main()
{
    //freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;
        char s[60];
        scanf("%d %s", &n, s);

        int a = 0, c = 0;
        for (int i = 0; i < n; i++)
        {
            if (s[i] == '(')
                a++;
            else
                a--;

            if (a < 0)
            {
                c++;
                a = 0;
            }
        }
        printf("%d\n", c);
    }

    return 0;
}