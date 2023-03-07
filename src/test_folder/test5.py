

ip_adress = ['172.16.0.1:36746', '172.16.0.1:54590', '172.16.0.1:45044', '172.16.0.1:42388', '172.16.0.1:61758', '172.16.0.1:54268', '172.16.0.1:43296', '172.16.0.1:51988']

res = {}

for x in ip_adress:
    # This represent the IP Address
    # print(x[0:x.index(':')])
    res[x[0:x.index(':')]] = sum(x[0:x.index(':')] in s for s in ip_adress)

print(res)