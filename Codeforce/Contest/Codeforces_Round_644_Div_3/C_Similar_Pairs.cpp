#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
    int test, n;

    scanf("%d", &test);

    while (test--)
    {
        scanf("%d", &n);
        int arr[50] = {};
        for (int i = 0; i < n; i++)
        {
            scanf("%d", &arr[i]);
        }

        sort(arr, arr + n);

        int flag = 0;
        int odd = 0, even = 0;

        for (int i = 0; i < n; ++i)
        {
            if (i != 0 && arr[i] - arr[i - 1] == 1)
            {
                flag = 1;
            }
            if (arr[i] % 2 == 1)
            {
                odd++;
            }
            else
            {
                even++;
            }
        }
        if (odd % 2 == 0 && even % 2 == 0)
        {
            printf("YES\n");
        }
        else if (odd % 2 == even % 2 && flag)
        {
            printf("YES\n");
        }
        else
        {
            printf("NO\n");
        }
    }
    return 0;
}