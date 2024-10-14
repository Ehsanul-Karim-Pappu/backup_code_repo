import math

n = int(input())
a = list(map(int, input().split()))
a.sort()
mx = a[n - 1]
r = int(math.ceil(mx ** (1 / float(n - 1))))
s1 = 0
k = 0
for i in a:
    if (i > r ** k):
        s1 += (i - r ** k)
    else:
        s1 += (r ** k - i)
    k += 1

# r = int(math.floor(mx ** (1 / float(n - 1))))
# s2 = 0
# k = 0
# for i in a:
#     if (i > r ** k):
#         s2 += (i - r ** k)
#     else:
#         s2 += (r ** k - i)
#     k += 1

# print(min(s1, s2))
print(s1)