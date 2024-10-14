#include <bits/stdc++.h>

using namespace std;

int main()
{
    //freopen("input.txt","r",stdin);

    int test;
    scanf("%d", &test);
    getchar();

    while (test--)
    {
        char n[10];
        int ans = 0;
        scanf("%s", n);

        if (strlen(n) == 1)
            ans += 1;
        if (strlen(n) == 2)
            ans += 3;
        if (strlen(n) == 3)
            ans += 6;
        if (strlen(n) == 4)
            ans += 10;

        int t = (int)n[0] - 49;
        // cout << t;

        ans += (t * 10);

        printf("%d\n", ans);
    }

    return 0;
}