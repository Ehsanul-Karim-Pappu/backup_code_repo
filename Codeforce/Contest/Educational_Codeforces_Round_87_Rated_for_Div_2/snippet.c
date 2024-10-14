#include <stdio.h>

int digit_sep(int number);
void swap(int *a, int *b);
void bubble_sort(int *array, int size);
int max(int *array, int size);
int min(int *array, int size);

//global variable
int digit[1000];

int main()
{
    //freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);

    int test, n;

    scanf("%d", &test);

    while (test--)
    {
        scanf("%d", &n);
    }

    return 0;
}

//functions

int digit_sep(int number)
{
    int size;
    for (size = 0; number != 0; size++)
    {
        digit[size] = number % 10;
        number /= 10;
    }
    return size;
}

void swap(int *a, int *b)
{
    int t;

    t = *b;
    *b = *a;
    *a = t;
}

void bubble_sort(int *array, int size)
{
    int i, j, flag;
    for (j = 0; j < size; j++)
    {
        flag = 0;
        for (i = 0; i < size - 1 - j; i++)
        {
            if (array[i] > array[i + 1])
            {
                swap(&array[i], &array[i + 1]);
                flag = 1;
            }
        }
        if (flag == 0)
            break;
    }
}

int max(int *array, int size)
{
    int i, mx;
    mx = array[0];

    for (i = 0; i < size; i++)
    {
        if (array[i] > mx)
            mx = array[i];
    }
    return mx;
}

int min(int *array, int size)
{
    int i, mn;
    mn = array[0];

    for (i = 0; i < size; i++)
    {
        if (array[i] < mn)
            mn = array[i];
    }
    return mn;
}