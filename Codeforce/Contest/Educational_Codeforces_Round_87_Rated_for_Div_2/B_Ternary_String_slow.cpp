#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    int test;

    scanf("%d", &test);
    getchar();

    while (test--)
    {
        int count[200000] = {0}, j = 0;
        char t, c, ch[200000];

        c = getchar();
        t = c;

        for (int i = 0; c != '\n'; i++)
        {
            if (t != c)
                j++;

            ch[j] = c;
            count[j]++;
            t = c;
            c = getchar();
            if (c == '\n')
                ch[j + 1] = '\0';
        }

        /*for (unsigned int i = 0; i < strlen(ch); i++)
        {
            printf("%c ", ch[i]);
        }
        printf("\n");
        for (unsigned int i = 0; i < strlen(ch); i++)
        {
            printf("%d ", count[i]);
        }
        printf("\n");*/

        int temp, ans = INT_MAX;

        for (unsigned int i = 1; i < strlen(ch) - 1; i++)
        {
            if (ch[i - 1] != ch[i + 1])
            {
                temp = count[i] + 2;
                if (temp < ans)
                {
                    ans = temp;
                    continue;
                }
            }
        }
        if (ans == INT_MAX)
            printf("0\n");
        else
            printf("%d\n", ans);
    }

    return 0;
}