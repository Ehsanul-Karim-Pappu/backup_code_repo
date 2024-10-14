#include <stdio.h>

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
    int test, n, k, i;

    scanf("%d", &test);

    while (test--)
    {
        scanf("%d %d", &n, &k);

        if (n < k)
            printf("NO\n");

        else if (n % 2 == 0)
        {
            if (n >= k * 2)
            {
                printf("YES\n");
                for (i = 1; i < k; i++)
                    printf("2 ");
                printf("%d", n - 2 * (k - 1));
                printf("\n");
            }
            else if (k % 2 == 1) //bijor songkhok bijor(jor songkha hobe na karon (n >= k * 2)=0) songkha jog korle jor hobe na
                printf("NO\n");
            else //i.e. 6 4 ans 1 1 1 3
            {
                printf("YES\n");
                for (i = 1; i < k; i++)
                    printf("1 ");
                printf("%d", n - (k - 1));
                printf("\n");
            }
        }
        else if (n % 2 == 1 && k % 2 == 1)
        {
            printf("YES\n");
            for (i = 1; i < k; i++)
                printf("1 ");
            printf("%d", n - (k - 1));
            printf("\n");
        }
        else if (n % 2 == 1 && k % 2 == 0) //i.e 97 2 jor songkhok bijor/jor jog korle bijor hobe na
            printf("NO\n");
    }
    return 0;
}