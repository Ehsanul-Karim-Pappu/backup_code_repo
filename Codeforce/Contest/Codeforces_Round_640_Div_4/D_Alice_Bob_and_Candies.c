#include <stdio.h>

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    int test, candy, alice, bob, a_index, b_index, c, a, b;

    scanf("%d", &test);

    while (test--)
    {
        alice = bob = 0;
        int flag = 1;
        c = 1;
        scanf("%d", &candy);
        int c_arr[1000] = {0};

        for (int i = 0; i < candy; i++)
            scanf("%d", &c_arr[i]);

        a_index = 0;
        b_index = candy - 1;
        alice = c_arr[a_index++];
        a = c_arr[0];
        b = 0;

        while (b_index >= a_index)
        {
            if (flag == 1)
            {
                flag = 0;
                while ((b <= a) && (b_index >= a_index))
                {
                    bob += c_arr[b_index];
                    b += c_arr[b_index];
                    b_index--;
                }
                c++;
                a = 0;
                //printf("alice %d/bob %d/b %d/c %d\n", alice, bob, b, c);
                continue;
            }
            else
            {
                flag = 1;
                while ((a <= b) && (b_index >= a_index))
                {
                    alice += c_arr[a_index];
                    a += c_arr[a_index];
                    a_index++;
                }
                c++;
                b = 0;
                //printf("alice %d/bob %d/a %d/c %d\n", alice, bob, a, c);
                continue;
            }
            a = b = 0;
        }

        printf("%d %d %d\n", c, alice, bob);
        //printf("--------------------------------------------------------------------\n");
    }

    return 0;
}