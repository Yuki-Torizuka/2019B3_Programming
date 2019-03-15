import numpy as np


def relu(x):
    return np.maximum(0, x)


class SingleLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        return relu(np.dot(self.W.T, x) + self.b)


x = np.array([1.0, 0.5])
W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b = np.array([0.1, 0.2, 0.3])

layer = SingleLayer(W, b)
print(layer.forward(x))
