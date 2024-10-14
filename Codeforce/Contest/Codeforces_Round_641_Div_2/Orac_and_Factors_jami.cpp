#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace std;

int smallestDiv(int n, int k)
{
    if (n % 2 == 0)
    {
        n += (2 * (k + 1));
        return n;
    }

    for (int i = 3; i * i <= n; i += 2)
    {
        if (n % i == 0)
            return i;
    }
    return n;
}

int main()
{
    int t, n, k;
    cin >> t;
    while (t--)
    {
        cin >> n >> k;
        while (k--)
        {
            if (n % 2 == 0)
            {
                n = smallestDiv(n, k);
                break;
            }
            else
            {
                n += smallestDiv(n, k);
            }
        }

        cout << n << endl;
    }
}