#include <stdio.h>
main()
{
    int target = 9, m, n;
    int nums[] = {2, 7, 11, 15};

    for (m = 0; m < 4; m++)
        for (n = m + 1; n < 4; n++)
            if ((nums[m] + nums[n]) == target)
                printf("[%d,%d]", m, n);
}