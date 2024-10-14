#include <bits/stdc++.h>

using namespace std;

int a[200000], w[200000];
int main()
{
    //freopen("input.txt", "r", stdin);

    int test;
    scanf("%d", &test);

    while (test--)
    {
        int n, k;
        scanf("%d %d", &n, &k);

        for (int i = 0; i < n; i++)
            scanf("%d", &a[i]);
        sort(a, a + n, greater<int>());

        for (int i = 0; i < k; i++)
            scanf("%d", &w[i]);
        sort(w, w + k, greater<int>());

        int l = 0, r = n - 1, i = k - 1;
        long long int ans = 0;
        while (w[i] == 1)
        {
            ans += (a[l++] * 2);
            i--;
        }

        for (int j = 0; j <= i; r -= (w[j] - 1), j++) //ekhane r -= (w[j] - 1), j++ er moddhe j++ age dile j++ operation age hobe
            ans += (a[l++] + a[r]);                   //w[j=0]-1=4-1=3, so new value of r=9-3=6, so a[r]=4 for input,
                                                      //1
                                                      //10 4
                                                      //10 9 8 7 6 5 4 3 2 1
                                                      //4 3 2 1

        printf("%lld\n", ans);
    }

    return 0;
}