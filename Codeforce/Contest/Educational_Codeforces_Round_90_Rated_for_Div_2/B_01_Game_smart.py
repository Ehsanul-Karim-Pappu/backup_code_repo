for i in range(int(input())):
    s = input()
    mn = min(s.count('1'), s.count('0'))
    if mn % 2:
        print("DA")
    else:
        print("NET")
