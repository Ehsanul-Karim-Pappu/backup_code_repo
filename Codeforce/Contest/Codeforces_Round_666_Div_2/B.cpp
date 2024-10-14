#include <bits/stdc++.h>

using namespace std;

int a[1000000];
int main()
{
    //freopen("input.txt","r",stdin);

    int n;
    scanf("%d", &n);

    for (int i = 0; i < n; i++)
        scanf("%d", &a[i]);

    sort(a, a + n);

    

    return 0;
}
