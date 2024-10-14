#include <bits/stdc++.h>

using namespace std;

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    int test;

    scanf("%d", &test);
    getchar();

    while (test--)
    {
        int j = 0;
        char t, c, ch[200000];

        vector<int> count;
        //count.clear();

        c = getchar();
        t = c;
        count.push_back(0);

        for (int i = 0; c != '\n'; i++)
        {
            if (t != c)
            {
                j++;
                count.push_back(1);
            }
            else
                count[j]++;

            ch[j] = c;
            t = c;
            c = getchar();
            if (c == '\n')
                ch[j + 1] = '\0';
        }

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