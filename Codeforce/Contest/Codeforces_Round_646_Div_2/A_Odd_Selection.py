for i in range(int(input())):
    n, x = map(int, input().split())
    a = list(map(int, input().split()))

    o = sum(i % 2 for i in a)
    e = n - o

    print("Yes" if (o >= x and (x % 2 or (not x % 2 and e >= 1))) or
          (o and o < x and (o % 2 or (not o % 2 and o - 1 + e >= x))) else "No")
