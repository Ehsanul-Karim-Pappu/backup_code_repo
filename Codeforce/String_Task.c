#include <stdio.h>
#include <string.h>

int main()
{
    int l, i;
    char ch[101];

    scanf("%s", &ch);
    l = strlen(ch);

    for (i = 0; i < l; i++)
    {
        if (ch[i] >= 'A' && ch[i] <= 'Z')
        {
            ch[i] = ch[i] + 32;
        }
    }

    for (i = 0; i < l; i++)
    {
        if (ch[i] != 'a' && ch[i] != 'o' && ch[i] != 'y' && ch[i] != 'e' && ch[i] != 'u' && ch[i] != 'i')
        {
            printf(".%c", ch[i]);
        }
    }
}
