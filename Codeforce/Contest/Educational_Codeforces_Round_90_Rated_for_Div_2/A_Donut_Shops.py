for i in range(int(input())):
    a, b, c = map(int, input().split())
    print(-1 if a >= c else 1, -1 if a * b <= c else b)
