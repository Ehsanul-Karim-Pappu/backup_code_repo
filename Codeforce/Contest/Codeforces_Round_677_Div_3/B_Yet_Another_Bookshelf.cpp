#include <bits/stdc++.h>

using namespace std;

int main()
{
    // freopen("input.txt","r",stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;
        scanf("%d", &n);
        getchar();

        char a[200];
        fgets(a, 200, stdin);

        int zero = 0, init = 0, fin = 0, ans = 0;
        for (int i = 0; a[i] != '\0'; i += 2)
        {
            if (a[i] == '1' && !init)
                init = i + 1;
            else if (a[i] == '1' && init)
            {
                fin = i;
                ans += zero;
                zero = 0;
            }
            else if (a[i] == '0' && init)
                zero++;
        }
        printf("%d\n", ans);
    }

    return 0;
}