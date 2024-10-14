#include <bits/stdc++.h>

using namespace std;

char a[500];
int main()
{
    // freopen("input.txt","r",stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;
        scanf("%d", &n);

        scanf("%s", a);

        for (int i = 0; i < 2 * n - 1; i++)
        {
            if (i % 2 == 0)
                printf("%c", a[i]);
        }
        printf("\n");
    }

    return 0;
}