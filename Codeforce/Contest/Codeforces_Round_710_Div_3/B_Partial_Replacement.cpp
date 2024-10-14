#include <bits/stdc++.h>

using namespace std;

char str[100];
int main()
{
    // freopen("input.txt","r",stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, k, ans = 0;
        scanf("%d %d", &n, &k);
        scanf("%s", str);

        bool flag = true;
        int first, last;
        for (int i = 0; i < n; i++)
        {
            if (str[i] == '*' && flag) 
            {
                first = i;
                flag = false;
            }

            if (str[i] == '*') last = i;
        }

        if (first == last)
        {
            printf ("1\n");
            continue;
        }
        else 
        {
            str[first] = 'x';
            str[last] = 'x';
            ans += 2;

            for (int i = first; i < last; )
            {
                if (i + k >= last) break;
                if(str[i + k] == '*')
                {
                    str[i + k] = 'x';
                    ans++;
                    i += k;
                }
                else
                {
                    for (int j = i + k - 1; i < j ; j--)
                    {
                        if (str[j] == '*')
                        {
                            str[j] = 'x';
                            ans++;
                            i = j;
                            break;
                        }
                    }
                }
            }
        }

        printf("%d\n", ans);
    }

    return 0;
}