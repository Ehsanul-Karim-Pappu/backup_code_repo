#include <bits/stdc++.h>

int min(int a, int b)
{
    if (a<=b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

int max(int a, int b)
{
    if (a>=b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
    int test;

    scanf("%d", &test);

    while (test--)
    {
        int a, b;
        scanf("%d %d", &a, &b);

        if (min(a, b) * 2 <= max(a, b))
        {
            printf("%d\n",max(a,b)*max(a,b) );
        }
        else if(min(a, b) * 2 > max(a, b))
        {
            printf("%d\n", min(a, b) * min(a, b)*4);
        }
        else if (a==b)
        {
            printf("%d\n", (a + b) * (a + b));
        }
        
        
    }

    return 0;
}