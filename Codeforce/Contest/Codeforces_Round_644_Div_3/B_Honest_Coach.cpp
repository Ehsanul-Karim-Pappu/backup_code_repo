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
        int arr[n];
        for (int i = 0; i < n; i++)
        {
            scanf("%d", &arr[i]);
        }

        sort(arr, arr + n);

        int diff = INT_MAX, temp = 0;

        for (int i = 1; i < n; i++)
        {
            temp = arr[i] - arr[i - 1];
            if (temp < diff)
            {
                diff = temp;
            }
        }
        printf("%d\n", diff);
    }

    return 0;
}