#include <stdio.h>

int main()
{
	freopen("input.txt", "r", stdin);
	//freopen("output.txt", "w", stdout);

	int test;

	scanf("%d", &test);

	while (test--)
	{
		int n, e = 0, mx = 0;

		scanf("%d", &n);
		int count[n + 1] = {0};

		for (int i = 0; i < n; i++)
		{
			scanf("%d", &e);
			count[e]++;
			if (e > mx)
				mx = e;
		}

		int c = 0, r = 0;

		for (int i = 1; i <= mx; i++)
		{
			if (count[i])
			{
				c += ((count[i] + r) / i);
				r = (count[i] + r) % i;
			}
		}

		printf("%d\n", c);
	}

	return 0;
}