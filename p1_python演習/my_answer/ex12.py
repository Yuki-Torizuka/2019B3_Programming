docs = []

with open("dataset/data.txt", "r", encoding="shift_jis") as f:
    for s_line in f:
        words = s_line.replace("\n", "").split("と")
        docs.append(words)

terms = sum(docs, [])
terms = list(set(terms))

print("docs: {}".format(docs))
print("terms: {}".format(terms))
