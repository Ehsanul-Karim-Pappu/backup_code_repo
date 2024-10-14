#include <bits/stdc++.h>

using namespace std;

vector<int> max_odd_divisor(int number)
{
    int mx = -1;
    for (int i = 2; i < (int)sqrt(number) + 1; i++)
    {
        if (number % i == 0)
        {
            if (i % 2 && i != 1)
                mx = max(mx, i);
            if ((number / i) % 2 == 1)
                mx = max(mx, number / i);
        }
    }
    vector<int> rt;
    rt.push_back(mx);
    rt.push_back(number / mx);
    return rt;
}

bool isPrime(int n)
{
    for (int i = 2; i < (int)sqrt(n) + 1; i++)
    {
        if (n % i == 0)
            return false;
    }
    return true;
}

int main()
{
    //freopen("input.txt", "r", stdin);

    int t;
    scanf("%d", &t);

    while (t--)
    {
        int n;
        scanf("%d", &n);

        if (n == 1)
            printf("FastestFinger\n");
        else if (n % 2 || n == 2)
            printf("Ashishgup\n");
        else
        {
            vector<int> a = max_odd_divisor(n);
            if (a[0] == -1 || (a[1] == 2 && isPrime(a[0])))
                printf("FastestFinger\n");
            else
                printf("Ashishgup\n");
        }
    }

    return 0;
}