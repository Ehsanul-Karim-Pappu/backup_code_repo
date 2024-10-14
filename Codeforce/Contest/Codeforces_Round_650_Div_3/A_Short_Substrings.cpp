#include <bits/stdc++.h>

using namespace std;

char str[100000];
int main()
{
    int t;
    scanf("%d", &t);
    getchar();

    while (t--)
    {
        scanf("%s", str);

        printf("%c", str[0]);
        for (unsigned int i = 1; i < strlen(str); i += 2)
            printf("%c", str[i]);

        printf("\n");
    }

    return 0;
}