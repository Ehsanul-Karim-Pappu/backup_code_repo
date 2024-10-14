#include <bits/stdc++.h>

using namespace std;

char s[100], t[100];

int main()
{
    // freopen("input.txt", "r", stdin);

    int q;
    scanf("%d", &q);

    while (q--)
    {
        int length_s, length_t, loop;
        int sim = 0, ans = 0, flag = 0;

        scanf("%s %s", s, t);
        length_s = strlen(s);
        length_t = strlen(t);

        loop = max(length_s, length_t);
        for (int i = 0; i < loop; i++)
        {
            if (s[i] == t[i])
            {
                sim++;
                flag = 1;
            }
            else
                break;
        }

        if (flag)
            ans = sim + 1 + length_s + length_t - 2 * sim;
        else
            ans = length_s + length_t;

        printf("%d\n", ans);
    }

    return 0;
}