#include <bits/stdc++.h>

using namespace std;

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, m, k, ans;
        double temp;

        scanf("%d %d %d", &n, &m, &k);

        if (m > n / k)
        {
            temp = ((m - (n / k)) * 1.) / (k - 1);
            ans = ceil(temp);
            printf("%d\n", (n / k) - ans);
        }
        else
        {
            printf("%d\n", m);
        }
    }
    return 0;
}