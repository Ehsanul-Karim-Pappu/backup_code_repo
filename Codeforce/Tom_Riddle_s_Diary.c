#include<stdio.h>
#include<string.h>

int main() {
	int n,i,j,count=0;
	char ch[101][101],str[101][101];

	scanf("%d",&n);
	j=0;
	for(i=0;i<n;i++){
		scanf("%s",&ch[i]);
		strcpy(str[j],ch[i]);
		j++;
	}

	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			if((strcmp(str[j],ch[i]))==0 && (i!=j)){
				count++;
				printf("%d\n",count);
			    //printf("YES\n");
			    break;
			}
			else{
				//printf("NO\n");
		        break;
			}
		}
	}

	if (count==1) printf("YES\n");
	else printf("NO\n");
}
