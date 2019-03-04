docs = [["リンゴ", "リンゴ"], ["リンゴ", "レモン"], ["レモン", "ミカン"]]
terms = ["リンゴ", "レモン", "ミカン"]


def tf(term, doc):
    count = 0
    for word in doc:
        if term == word:
            count += 1

    return count / len(doc)


for doc in docs:
    for term in terms:
        print("tf({0}, {1}) = {2} ".format(term, doc, tf(term, doc)), end="")
