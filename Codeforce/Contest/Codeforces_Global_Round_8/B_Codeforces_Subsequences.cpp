#include <bits/stdc++.h>

using namespace std;

int main()
{
    long long int k;
    scanf("%lld", &k);

    string str = "codeforces";
    int n = (int)str.size();
    vector<int> v(n, 1);
    long long int ans = 1;
    bool flag = false;

    while (1)
    {
        for (int i = 0; i < n; i++)
        {
            if (ans >= k)
            {
                flag = true;
                break;
            }
            ans /= v[i];
            v[i]++;
            ans *= v[i];
        }
        if (flag)
            break;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < v[i]; j++)
            printf("%c", str[i]);
    }

    return 0;
}