#include <stdio.h>
#include <math.h>
int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
    int t, n, ans;
    scanf("%d", &t);
    while (t--)
    {
        scanf("%d", &n);
        for (int i = 2; i <= 30; i++)
        {
            int p = pow(2, i) - 1;
            if (n % p == 0)
            {
                ans = n / (pow(2, i) - 1);
                printf("%d\n", ans);
                break;
            }
        }
    }
}