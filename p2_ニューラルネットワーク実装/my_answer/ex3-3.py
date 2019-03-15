import numpy as np


class Softmax():
    def __int__(self):
        self.z = None

    def forward(self, x):
        self.z = 1. / (1. + np.exp(-x))
        return self.z

    def backprop(self, dz):
        dx = self.z * (1. - self.z)
        return dx


x = np.array([-0.5, 0.0, 1.0, 2.0])
sigmoid = Softmax()

print("順伝播出力: {0}".format(sigmoid.forward(x)))
print("逆伝播出力: {0}".format(sigmoid.backprop(1.)))
