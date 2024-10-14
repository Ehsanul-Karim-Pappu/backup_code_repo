#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt","r",stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;
        scanf("%d", &n);

        int t = n;
        while (t--)
            printf("%d ", n);
        printf("\n");
    }

    return 0;
}