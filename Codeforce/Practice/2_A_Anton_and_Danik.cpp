#include <bits/stdc++.h>

using namespace std;

int main()
{
    int n;
    scanf("%d", &n);

    char s[100000];
    scanf("%s", s);

    int a = 0, d = 0;
    for (int i = 0; i < n; i++)
    {
        if (s[i] == 'A')
            a++;
        else
            d++;
    }

    if (a > d)
        printf("Anton\n");
    else if (a < d)
        printf("Danik\n");
    else
        printf("Friendship\n");

    return 0;
}