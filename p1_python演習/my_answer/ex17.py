import numpy as np
from nlp.ex15 import tf_idf
from nlp.ex16 import cosine_sim

docs = []
terms = []

with open("dataset/data.txt", "r", encoding="shift_jis") as f:
    for s_line in f:
        words = s_line.replace("\n", "").split("と")
        docs.append(words)

terms = sum(docs, [])
terms = list(set(terms))

tfidf = tf_idf(terms, docs)
cosine = np.zeros([len(docs), len(docs)])

for doc1 in range(len(docs)):
    for doc2 in range(len(docs)):
        cosine[doc1, doc2] = cosine_sim(tfidf[doc1], tfidf[doc2])

print(cosine)
