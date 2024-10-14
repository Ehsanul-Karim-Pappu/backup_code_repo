#include<bits/stdc++.h>

using namespace std;

int a[1000000];

int main()
{
    // freopen("input.txt","r",stdin);

    int n, q, t, p, ones = 0;
    scanf("%d %d", &n, &q);

    for(int i = 0; i < n; i++)
    {
        scanf("%d", &a[i]);
        if (a[i] == 1) ones++;
    }
        // printf("ones = %d\n", ones);

    for(int i = 0; i < q; i++)
    {
        scanf("%d %d", &t, &p);

        if (t == 1)
        {
            a[p - 1] = 1 - a[p - 1];
            if (a[p - 1] == 0) ones--;
            else ones++;
            // printf("ones-- = %d\n", ones);
        }
        else 
        {
            // printf("ones = %d\n", ones);
            if (p > ones) printf("0\n");
            else printf("1\n");
        }
    }

    return 0;
}