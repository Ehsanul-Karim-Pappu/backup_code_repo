//#include <stdio.h>
//#include <algorithm>
//#include <vector>
#include <bits/stdc++.h>

using namespace std;

#define ll long long int

vector<ll> factors(ll x)
{
	vector<ll> result;
	ll i = 2;

	while (i * i <= x)
	{
		if (x % i == 0 && i != x)
		{
			result.push_back(i);
			if (x / i != i)
				result.push_back(x / i);
		}
		i++;
	}

	return result;
}

bool isPrime(ll n)
{
	if (n < 2)
		return false;

	for (ll i = 2; i * i <= n; i++)
	{
		if (n % i == 0)
			return false;
	}

	return true;
}

int main()
{
	freopen("input.txt", "r", stdin);
	//freopen("output.txt", "w", stdout);

	ll test;
	scanf("%lld", &test);

	while (test--)
	{
		vector<ll> ans;
		ll n, k, i;
		scanf("%lld %lld", &n, &k);

		if (n <= k)
			printf("1\n");
		else if (isPrime(n) || k == 1)
			printf("%lld\n", n);
		else
		{
			for (i = ans.size() - 1; i >= 0; i--)
			{
				if (n % ans[i] == 0)
					break;
			}
			printf("%lld\n", n / ans[i]);
		}
	}

	return 0;
}