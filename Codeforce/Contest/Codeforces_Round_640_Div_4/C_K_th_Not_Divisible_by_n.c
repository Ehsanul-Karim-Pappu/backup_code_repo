#include <stdio.h>
int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
    int test, s, ans;
    double n, k, i;

    scanf("%d", &test);

    while (test--)
    {
        scanf("%lf %lf", &n, &k);
        i = k / (n - 1);
        if (i > (int)i)
        {
            i = (int)i;
            i++;
        }
        s = n * i;
        ans = (s - 1) - (((n - 1) * i) - k);
        printf("%d\n", ans);
    }

    return 0;
}