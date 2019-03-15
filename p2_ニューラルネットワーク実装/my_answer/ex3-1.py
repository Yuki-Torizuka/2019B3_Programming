import numpy as np


class Multiply():
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        z = x * y
        return z

    def backprop(self, dz):
        dx = dz * self.y
        dy = dz * self.x
        return dx, dy


class Add():
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        z = x + y
        return z

    def backprop(self, dz):
        dx = dz
        dy = dz
        return dx, dy


a = 2
b = 3
c = 4

add = Add()
mult = Multiply()

o1 = add.forward(a, b)
o2 = mult.forward(o1, c)

print("順伝播出力: {0}".format(o2))

do1, dc = mult.backprop(1)
da, db = add.backprop(do1)

print("逆伝播出力 da: {0}, db: {1}, dc: {2}".format(da, db, dc))
