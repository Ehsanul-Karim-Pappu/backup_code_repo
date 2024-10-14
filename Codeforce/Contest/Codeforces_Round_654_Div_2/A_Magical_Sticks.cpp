#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;
        scanf("%d", &n);

        double ans = ceil(n * 1. / 2);

        printf("%d\n", (int)ans);
    }

    return 0;
}