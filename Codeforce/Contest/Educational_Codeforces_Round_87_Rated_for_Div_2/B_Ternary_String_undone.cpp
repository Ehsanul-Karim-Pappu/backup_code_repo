#include <stdio.h>
#include <string.h>

int main()
{
    freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    int test, i, count[2001], c, j, a, k;
    char str[] = "123", input[2001];

    scanf("%d", &test);
    getchar();

    while (test--)
    {
         a = 0;
        scanf("%s", input);

        for (k = 0; a <= strlen(input) - 3; k++)
        {
            c = 0;
            for (i = 0; i < 3; i++)
            {
                for (j = a; input[j] != '\0'; j++)
                {
                    c++;
                    if (str[i] == input[j])
                    {
                        a = j + 1;
                        break;
                    }
                }
            }
            count[k] = c;
            j--;
            a--;
            while (input[j] != input[a])
            {
                j--;
                a--;
            }
        }
    }

    return 0;
}