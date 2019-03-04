import numpy as np

docs = [["リンゴ", "リンゴ"], ["リンゴ", "レモン"], ["レモン", "ミカン"]]
terms = ["リンゴ", "レモン", "ミカン"]


def tf(term, doc):
    count = 0

    for word in doc:
        if term == word:
            count += 1

    return count / len(doc)


def idf(term, docs):
    count = 0

    for doc in docs:
        if term in doc:
            count += 1

    return np.log10(len(docs) / count) + 1.


def tf_idf(terms, docs):
    tfidf = np.zeros([len(docs), len(terms)])

    for x, doc in enumerate(docs):
        for y, term in enumerate(terms):
            tfidf[x, y] = tf(term, doc) * idf(term, docs)

    return tfidf


print(tf_idf(terms, docs))

