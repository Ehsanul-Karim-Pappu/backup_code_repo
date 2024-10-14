#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        long long int a, b;
        scanf("%lld %lld", &a, &b);

        long long int temp, c = 0;
        if (a == b)
        {
            printf("0\n");
            continue;
        }
        else if (max(a, b) % min(a, b) != 0)
        {
            printf("-1\n");
            continue;
        }
        else
        {

            temp = max(a, b) / min(a, b);
            while (temp >= 8)
            {
                if (temp % 8 != 0)
                    break;
                temp /= 8;
                c++;
            }
            while (temp >= 4)
            {
                if (temp % 4 != 0)
                    break;
                temp /= 4;
                c++;
            }
            while (temp >= 2)
            {
                if (temp % 2 != 0)
                    break;
                temp /= 2;
                c++;
            }
        }
        if (temp == 1)
            printf("%lld\n", c);
        else
            printf("-1\n");
    }

    return 0;
}