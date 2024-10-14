#include <bits/stdc++.h>

using namespace std;

int main()
{
    //freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        long long int a, b, c;
        scanf("%lld %lld %lld", &a, &b, &c);

        printf(a >= c ? "-1 " : "1 ");
        printf(a * b <= c ? "-1\n" : "%lld\n", b);
        //printf("%d %d\n", a >= c ? -1 : 1, a <= c / b ? -1 : b);
    }

    return 0;
}