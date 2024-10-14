#include <bits/stdc++.h>

using namespace std;

int main()
{
    freopen("input.txt", "r", stdin);
    
    int t;
    cin >> t;

    while (t--)
    {
        int a, b;
        cin >> a >> b;

        if (a > b)
            swap(a, b);

        cout << min((a + b) / 3, a) << endl;
    }
    return 0;
}