// wrong for case 1 1 2 2 3 6

#include <stdio.h>
#include <algorithm>

using namespace std;

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
    int test, n;

    scanf("%d", &test);

    while (test--)
    {
        scanf("%d", &n);
        int arr[51] = {};
        for (int i = 0; i < n; i++)
        {
            scanf("%d", &arr[i]);
        }

        sort(arr, arr + n);

        for (int i = 0; i < n - 1 && arr[i] != 0; i++)
        {
            for (int k = 0; k < n; k++)
            {
                if (arr[k + 1] != arr[k])
                    break;
            }
            if (arr[i + 1] - arr[i] == 1)
            {
                arr[i] = arr[i + 1] = 0;
                i++;
            }
            else if (arr[i + 1] == arr[i])
            {
                if (arr[i + 2] - arr[i] == 1)
                    arr[i + 2] = arr[i] == 0;
            }
            else
            {
                if ((arr[i] % 2 == 0) && arr[i] != 0)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        if (arr[j] % 2 == 0 && arr[j] != 0)
                        {
                            arr[j] = arr[i] = 0;
                            if (j == i + 1)
                            {
                                i++;
                            }
                            break;
                        }
                    }
                }
                else if ((arr[i] % 2 == 1) && arr[i] != 0)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        if (arr[j] % 2 == 1 && arr[j] != 0)
                        {
                            arr[j] = arr[i] = 0;
                            if (j == i + 1)
                            {
                                i++;
                            }
                            break;
                        }
                    }
                }
            }
        }
        int flag = 0;
        for (int i = 0; i < n; i++)
        {
            if (arr[i] != 0)
            {
                flag = 1;
                break;
            }
        }
        if (flag == 1)
        {
            printf("NO\n");
        }
        else
        {
            printf("YES\n");
        }
    }

    return 0;
}