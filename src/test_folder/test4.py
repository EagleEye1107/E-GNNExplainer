


x = 2300825

# print(int(x/4))

a = 0
b = -1

for i in range(2300825):
    if i != 0 :
        if i % int(x/4) == 0:
            print(i)
            a = b + 1
            b = i
            print(f"[{a}, {b}]")