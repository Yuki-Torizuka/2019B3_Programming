s = "Hi He Lead Because Boron Could Not Oxidize Flourine. New Nations Might Also Sign Peace Security Clause. " \
    "Arthur King Can."

dic = {}
words = s.replace(",", "").split()

numbers = [1, 5, 6, 7, 8, 9, 15, 16, 19]
count = 0

for i in words:
    count += 1
    if count in numbers:
        dic[i[:1]] = count
    else:
        dic[i[:2]] = count

print(dic)
