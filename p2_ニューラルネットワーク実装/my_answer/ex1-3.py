import numpy as np


class Perceptron:
    def __init__(self, w1, w2, theta):
        self.w1 = w1
        self.w2 = w2
        self.theta = theta

    def forward(self, x1, x2):
        if (self.w1 * x1 + self.w2 * x2) >= self.theta:
            return 1
        else:
            return 0


x1_list = [1, 1, 0, 0]
x2_list = [1, 0, 1, 0]

and_gate = Perceptron(1, 1, 1.5)
nand_gate = Perceptron(-1, -1, -1.5)
or_gate = Perceptron(1, 1, 0.5)


def xor(x1, x2):
    return and_gate.forward(nand_gate.forward(x1, x2), or_gate.forward(x1, x2))


for x1, x2 in zip(x1_list, x2_list):
    print("XOR({0}, {1}) = {2} ".format(x1, x2, xor(x1, x2)))
