#include <stdio.h>

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int m, n;
        scanf("%d %d", &m, &n);

        double ans = ((m * n) * 1.) / 2;

        if (ans > (m * n) / 2)
        {
            int result = ((m * n) / 2) + 1;
            printf("%d\n", result);
        }
        else
            printf("%d\n", (m * n) / 2);
    }
    return 0;
}