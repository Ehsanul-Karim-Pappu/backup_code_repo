#include <stdio.h>
#include <math.h>

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    int t, n, m, ans;
    scanf("%d", &t);
    while (t--)
    {
        scanf("%d %d", &n, &m);

        if (n > 2)
        {
            printf("%d\n", 2 * m);
        }
        else if (n == 1)
        {
            printf("%d\n", 0);
        }
        else if (n == 2)
        {
            printf("%d\n", m);
        }
    }
    return 0;
}