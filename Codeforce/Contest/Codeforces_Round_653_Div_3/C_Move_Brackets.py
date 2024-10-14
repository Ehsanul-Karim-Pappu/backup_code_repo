def checkBalance(mystr):
    arr = []
    i = 0
    for mystr[i] in mystr:
        if mystr[i] == '(':
            arr.append(mystr[i])
        elif not arr and mystr[i] == ')':
            return False
        elif i and mystr[i] == ')' and arr[-1] == '(':
            arr.pop()
        i += 1

    if arr:
        return False
    else:
        return True


for i in range(int(input())):
    n = int(input())
    s = list(input())

    c = 0
    while not checkBalance(s):
        c += 1
        i = 0
        for s[i] in s:
            if s[i] == ")":
                b = s.pop(i)
                s.append(b)
                break
            i += 1
    else:
        print(c)
