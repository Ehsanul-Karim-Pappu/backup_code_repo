#include <stdio.h>

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    int n, s;

    scanf("%d %d", &n, &s);

    if (s < n * 2)
        printf("NO\n");
    else
    {
        printf("YES\n");
        for (int i = 0; i < n - 1; i++)
            printf("2 ");
        printf("%d\n", s - (2 * (n - 1)));
        printf("1\n");
    }

    return 0;
}