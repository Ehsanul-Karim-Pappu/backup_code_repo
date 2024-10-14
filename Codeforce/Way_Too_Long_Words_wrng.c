#include<stdio.h>
#include<string.h>

int main() {
    int n,l,i;
    char ch[100][100];
    scanf("%d",&n);
    for(i=0; i<n; i++) {
        scanf("%s",&ch[i]);
    }
    printf("%s\n",ch[0]);


    /*l=strlen(ch);
    if (l>10)
        printf("%c%d%c",ch[0],l-2,ch[l-1]); */


    return 0;
}
