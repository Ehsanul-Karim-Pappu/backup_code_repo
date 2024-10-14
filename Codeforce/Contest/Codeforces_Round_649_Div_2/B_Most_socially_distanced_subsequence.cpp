#include <bits/stdc++.h>

using namespace std;

int p[100000];
vector<int> find_large(int position, int size)
{
    int update = p[position];
    vector<int> result;
    for (int i = position + 1; i < size; i++)
    {
        if (p[i] > update)
        {
            result.push_back(i - 1);
            result.push_back(update);
            return result;
        }
        else
            update = p[i];
    }
    result.push_back(size - 1);
    result.push_back(p[size - 1]);
    return result;
}

vector<int> find_small(int position, int size)
{
    int update = p[position];
    vector<int> result;
    for (int i = position + 1; i < size; i++)
    {
        if (p[i] < update)
        {
            result.push_back(i - 1);
            result.push_back(update);
            return result;
        }
        else
            update = p[i];
    }
    result.push_back(size - 1);
    result.push_back(p[size - 1]);
    return result;
}

int main()
{
    //freopen("input.txt", "r", stdin);

    int t;
    scanf("%d", &t);

    while (t--)
    {
        int n;
        scanf("%d", &n);

        for (int i = 0; i < n; i++)
            scanf("%d", &p[i]);

        vector<int> temp, ans;
        temp.clear();
        ans.clear();
        for (int i = 0; i < n; i++)
        {
            if (!i)
                ans.push_back(p[i]);
            else if (p[i] < p[i - 1])
            {
                temp = find_large(i, n);
                ans.push_back(temp[1]);
                i = temp[0];
                if (temp[1] == p[n - 1])
                    break;
            }
            else if (p[i] > p[i - 1])
            {
                temp = find_small(i, n);
                ans.push_back(temp[1]);
                i = temp[0];
                if (temp[1] == p[n - 1])
                    break;
            }
        }

        printf("%d\n", ans.size());
        for (int i = 0; i < ans.size(); i++)
            printf("%d ", ans[i]);
        printf("\n");
    }

    return 0;
}