#include <bits/stdc++.h>

using namespace std;

char s[110];
int main()
{
    int test;
    scanf("%d", &test);

    while (test--)
    {
        scanf("%s", s);

        int one = 0, zero = 0;
        for (int i = 0; s[i] != '\0'; i++)
        {
            if (s[i] == '1')
                one++;
            else
                zero++;
        }

        printf(min(one, zero) % 2 ? "DA\n" : "NET\n");
    }

    return 0;
}