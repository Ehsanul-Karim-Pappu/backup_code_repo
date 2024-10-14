#include <bits/stdc++.h>

using namespace std;

char s[1000000];
int main()
{
    // freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, cnt[30] = {0};
        scanf("%d", &n);

        for (int i = 0; i < n; i++)
        {
            scanf("%s", s);
            for (int i = 0; s[i] != '\0'; i++)
            {
                cnt[(int)s[i] - 96]++;
            }
        }

        bool flag = false;
        for (int i = 1; i <= 28; i++)
        {
            if (cnt[i] % n)
            {
                flag = true;
                break;
            }
        }
        if (flag)
            printf("NO\n");
        else
            printf("YES\n");
    }

    return 0;
}