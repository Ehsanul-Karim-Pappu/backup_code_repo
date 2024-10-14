#include <stdio.h>

int bubble_sort(int ch[], int size)
{
    int i, j, flag, s;
    for (j = 0; j < size; j++)
    {
        flag = 0;
        for (i = 0; i < size - 1 - j; i++)
        {
            if (ch[i] > ch[i + 1])
            {
                s = ch[i];
                ch[i] = ch[i + 1];
                ch[i + 1] = s;
                flag = 1;
            }
        }
        if (flag == 0)
            break;
    }
    return ch[i];
}

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    int n[55], i, j;
    char ch = '+';

    for (i = 0; i < 55; i++)
        n[i] = 4;

    for (i = 0; ch != '\n'; i++)
    {
        scanf("%d", &n[i]);
        ch = getchar();
    }

    n[i] = bubble_sort(n, i);

    for (j = 0; j < i; j++)
    {
        printf("%d", n[j]);
        if (j == i - 1)
            printf("\n");
        else
            printf("+");
    }

    return 0;
}