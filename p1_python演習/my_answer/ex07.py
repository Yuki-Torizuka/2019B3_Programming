s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

words = s.replace(",", "").replace(".", "").split()

words_len = []
for i in words:
    words_len.append(len(i))

print(words_len)
