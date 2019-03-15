import numpy as np


def softmax(x):
    max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - max)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x


class SoftmaxCrossEntropy:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        return np.average(np.sum(-self.t * np.log(self.y), axis=1))

    def backprop(self, dz):
        return softmax(x) - self.t


x = np.array([[1.0, 0.5], [-0.4, 0.1]])
t = np.array([[1.0, 0.0], [0.0, 1.0]])

sce = SoftmaxCrossEntropy()

print("順伝播出力:\n{0}".format(sce.forward(x, t)))
print("逆伝播出力:\n{0}".format(sce.backprop(1)))
