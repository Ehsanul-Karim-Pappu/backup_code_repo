#include <bits/stdc++.h>

using namespace std;

char s[100010];
int main()
{
    //freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;
        scanf("%d", &n);
        getchar();

        bool flag = false;
        for (int i = 0; i < n; i++)
        {
            scanf("%c", &s[i]);
            if (s[i] == '1')
                flag = true;
            if (!flag)
                printf("0");
        }

        bool one = false;
        int c = 0;
        for (int i = 0; i < n; i++)
        {
            if (s[i] == '1')
                one = true;

            if (one && s[i] == '0')
            {
                one = false;
                c++;
            }
        }

        if (c != 0)
            printf("0");

        for (int i = n - 1; i >= 0; i--)
        {
            if (s[i] == '0')
                break;
            else
                printf("1");
        }
        printf("\n");
    }

    return 0;
}