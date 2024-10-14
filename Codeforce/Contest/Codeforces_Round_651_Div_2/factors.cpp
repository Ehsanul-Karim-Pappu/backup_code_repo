#include <bits/stdc++.h>

using namespace std;

vector<int> factors(int number)
{
    vector<int> divisor;
    divisor.clear();
    for (int i = 1; i < (int)sqrt(number) + 1; i++)
    {
        if (number % i == 0)
        {
            divisor.push_back(i);
            divisor.push_back(number / i);
        }
    }
    vector<int> bak = divisor;
    sort(bak.begin(), bak.end());

    return divisor;
}

int main()
{
    int n;
    scanf("%d", &n);
    vector<int> disp;
    //for (int i = 1; i <= n; i++)
    //{
    disp = factors(n);
    for (int j = 0; j < (int)disp.size(); j += 2)
        printf("%d X %d = %d\n", disp[j], disp[j + 1], n);
    printf("\n");
    //}

    return 0;
}