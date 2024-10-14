#include<stdio.h>
#include<string.h>

int main() {
    int n,l,i;
    char ch[101][101];
    scanf("%d",&n);
    for(i=0; i<n; i++) {
        scanf("%s",&ch[i]);
    }

    l=strlen(ch);
    if (l>10)
        printf("%c%d%c",ch[0],l-2,ch[l-1]);


    return 0;
}
