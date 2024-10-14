#include <stdio.h>

int main()
{
    int t;
    scanf("%d", &t);
    while (t--)
    {
        int n, k, gold = 0, count = 0;
        scanf("%d", &n);
        scanf("%d", &k);

        int a[200];
        for (int i = 0; i < n; i++)
        {
            scanf("%d", &a[i]);
            if (a[i] >= k)
            {
                gold = gold + a[i];
            }
            if (a[i] == 0 && gold != 0)
            {
                gold--;
                count++;
            }
        }
        printf("%d\n", count);
    }
}