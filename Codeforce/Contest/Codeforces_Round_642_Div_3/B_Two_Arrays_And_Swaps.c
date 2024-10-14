#include <stdio.h>

int max(int *arr, int size)
{
	int mx, i;
	mx = arr[0];

	for (i = 0; i < size; i++)
	{
		if (arr[i] > mx)
			mx = arr[i];
	}
	return mx;
}

int min(int *arr, int size)
{
	int mn, i;
	mn = arr[0];

	for (i = 0; i < size; i++)
	{
		if (arr[i] < mn)
			mn = arr[i];
	}
	return mn;
}

void swap(int *a, int *b)
{
	int t;

	t = *b;
	*b = *a;
	*a = t;
}

int main()
{
	//freopen("input.txt", "r", stdin);
	//freopen("output.txt", "w", stdout);

	int t, n, k, i;
	scanf("%d", &t);
	while (t--)
	{
		scanf("%d %d", &n, &k);
		int a[n], b[n], x[n], y[n];

		for (i = 0; i < n; i++)
			scanf("%d", &a[i]);
		for (i = 0; i < n; i++)
			scanf("%d", &b[i]);

		while (k--)
		{
			i = 0;
			if (min(a, n) < max(b, n) && min(a, n) > 0 && max(b, n) < 31)
			{
				swap(min(a, n), max(b, n));
				x[i] = max(b, n);
				i++;
			}
			else if(min(a, n) > 0 && max(b, n) < 31)
			{
				x[i] = a[i];
				y[i] = b[i];
				i++;
			}
			for (i = 0; i < n; i++)
				printf("%d ", x[i]);
		}
	}
	return 0;
}