#include <bits/stdc++.h>

using namespace std;

int isPerfectSquare(int n)
{
    if (n % 2)
    {
        for (int i = 3; i <= (int)sqrt(n); i += 2)
        {
            while (n % i == 0)
                n /= i;
            if (n == 1)
                return i;
        }
        return 0;
    }
    else
        return 0;
}


bool isPrime(int n)
{
    // Corner cases
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;

    // This is checked so that we can skip
    // middle five numbers in below loop
    if (n % 2 == 0 || n % 3 == 0)
        return false;

    for (int i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;

    return true;
}

int main()
{
    freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n;
        scanf("%d", &n);
        if (n % 2 == 0)
            printf("%d %d\n", n / 2, n / 2);
        else if (isPrime(n))
            printf("%d %d\n", 1, n - 1);
        else if (isPerfectSquare(n))
            printf("%d %d\n", n / isPerfectSquare(n), n - n / isPerfectSquare(n));
        else
            printf("%d %d\n", n / 3, n - n / 3);
    }

    return 0;
}