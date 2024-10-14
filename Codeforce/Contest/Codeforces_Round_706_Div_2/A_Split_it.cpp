#include <bits/stdc++.h>

using namespace std;

char str[200];
int main()
{
    freopen("input.txt","r",stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, k;
        bool flag = false;
        scanf("%d %d", &n, &k);

        scanf("%s", str);

        int j = n - 1;
        for (int i = 0; i <= k - 1; i++)
        {
            if (str[i] != str[j])
                flag = true;
            j--;
        }

        if (flag || 2 * k >= n)
            printf("NO\n");
        else
            printf("YES\n");
    }

    return 0;
}