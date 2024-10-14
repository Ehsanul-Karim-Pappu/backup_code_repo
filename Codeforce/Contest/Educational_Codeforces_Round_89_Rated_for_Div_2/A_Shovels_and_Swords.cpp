#include <bits/stdc++.h>

using namespace std;

int main()
{
    //freopen("input.txt", "r", stdin);

    int t;
    scanf("%d", &t);
    while (t--)
    {
        int a, b;
        scanf("%d %d", &a, &b);

        if (min(a, b) * 2 <= max(a, b))
        {
            printf("%d\n", min(a, b));
            continue;
        }
        else
        {
            printf("%d\n", (a + b) / 3);
            continue;
        }
    }

    return 0;
}