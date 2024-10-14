test = int(input())

for i in range(test):
    s = list(input())
    if len(s) == 1:
        print("NET")
        continue
    i = 0
    alice = True
    while True:
        flag = False
        if s[i] != s[i + 1]:
            flag = True
            if alice:
                s.remove(s[i])
                s.remove(s[i])
                alice = False
            else:
                s.remove(s[i])
                s.remove(s[i])
                alice = True
        if not flag:
            i += 1
        else:
            i = 0

        if i + 1 >= len(s):
            break

    if alice:
        print("NET")
    else:
        print("DA")
