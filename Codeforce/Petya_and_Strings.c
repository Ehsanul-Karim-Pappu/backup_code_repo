    #include<stdio.h>
    #include<string.h>

    int main(){
        char ch[101], str[101];
        int i,j=0,a=0,b=0;

        scanf("%s",&ch);
        scanf("%s",&str);

        for(i=0;i<strlen(ch);i++){
            if (ch[i]>='A'&&ch[i]<='Z'){
                ch[i]=ch[i]+32;
            }
        }
        for(i=0;i<strlen(str);i++){
            if (str[i]>='A' && str[i]<='Z'){
                str[i]=str[i]+32;
            }
        }

        for(i=0;i<strlen(ch);i++){
            if(ch[i]<str[j]) {a++; break;}
            else if(ch[i]>str[j]) {b++; break;}
            j++;
        }
        if (a<b) printf("1");
        else if (a==b) printf("0");
        else printf("-1");
    }
