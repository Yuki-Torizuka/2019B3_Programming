import random
import sys

shuffles = []
for word in sys.argv[1::]:
    if len(word) > 3:
        arr = list(word[1:-1])
        random.shuffle(arr)
        new_word = word[0] + "".join(arr) + word[-1]
    else:
        new_word = word
    shuffles.append(new_word)

print(" ".join(shuffles))


