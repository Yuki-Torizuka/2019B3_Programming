# coding:utf-8
import sys


def n_gram(s, n):
    gram = []
    for i in range(len(s) - n + 1):
        gram.append(s[i:i + n])
    return gram


print("単語bi-gram: {0}".format(n_gram(sys.argv[1:], 2)))
print("文字bi-gram: {0}".format(n_gram("".join(sys.argv[1:]), 2)))
