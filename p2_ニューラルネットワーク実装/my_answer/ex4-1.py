import numpy as np
from collections import OrderedDict

class ReLU:
    def __int__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0).astype(np.int)
        return x * self.mask

    def backprop(self, dz):
        return dz * self.mask


class Affine:
    def __init__(self):
        self.W = None
        self.b = None
        self.x = None

    def forward(self, x, W, b):
        self.x = x
        self.W = W
        self.b = b
        return np.dot(x, self.W) + self.b

    def backprop(self, dz):
        dx = np.dot(dz, self.W.T)
        dW = np.dot(self.x.T, dz)
        db = np.sum(dz, axis=0)
        return dx, dW, db

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

    def backprop(self):
        return self.y - self.t


class NeuralNetwork:
    def __init__(self):
        self.parameters_name = ["W1", "b1", "W2", "b2"]
        self.parameters = {}
        self.parameters_grad = {}
        self.parameters["W1"] = np.random.normal(0., 0.1, [784, 500])
        self.parameters["b1"] = np.zeros([500])
        self.parameters["W2"] = np.random.normal(0, 0.1, [500, 10])
        self.parameters["b2"] = np.zeros([10])

        self.affine1 = Affine()
        self.relu1 = ReLU()
        self.affine2 = Affine()
        self.sce = SoftmaxCrossEntropy()

    def forward(self, x):
        z1 = self.affine1.forward(x, self.parameters["W1"], self.parameters["b1"])
        z2 = self.relu1.forward(z1)
        z3 = self.affine2.forward(z2, self.parameters["W2"], self.parameters["b2"])
        return z3

    def loss(self, x, t):
        return self.sce.forward(self.forward(x), t)

    def backprop(self, x, t):
        self.loss(x, t)
        dz3 = self.sce.backprop()
        dz2, self.parameters_grad["W2"], self.parameters_grad["b2"] = self.affine2.backprop(dz3)
        dz1 = self.relu1.backprop(dz2)
        _, self.parameters_grad["W1"], self.parameters_grad["b1"] = self.affine1.backprop(dz1)

    def sgd(self, x, t):
        learning_rate = 0.0001
        self.backprop(x, t)
        for parameter in self.parameters_name:
            self.parameters[parameter] -= learning_rate * self.parameters_grad[parameter]
