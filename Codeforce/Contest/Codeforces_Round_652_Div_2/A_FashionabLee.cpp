#include <bits/stdc++.h>

using namespace std;

int main()
{
    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;
        scanf("%d", &n);
        if (n % 4 == 0)
            printf("Yes\n");
        else
            printf("No\n");
    }

    return 0;
}