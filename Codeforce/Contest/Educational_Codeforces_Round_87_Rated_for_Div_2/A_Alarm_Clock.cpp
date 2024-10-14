#include <bits/stdc++.h>

using namespace std;

int main()
{
	freopen("input.txt", "r", stdin);
	//freopen("output.txt", "w", stdout);

	long long test, ans;
	double a, b, c, d, count;
	scanf("%lld", &test);

	while (test--)
	{
		count = 0;
		scanf("%lf %lf %lf %lf", &a, &b, &c, &d);

		if (a <= b)
		{
			printf("%lld\n", b);
			continue;
		}
		else
		{
			if (c <= d)
			{
				printf("-1\n");
				continue;
			}
			else
			{
				count = (a - b) / (c - d);
				count = ceil(count);
			}
		}
		ans = b + count * c;
		printf("%lld\n", ans);
	}

	return 0;
}