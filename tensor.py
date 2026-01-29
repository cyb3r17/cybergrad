import math
import cupy as cp

class Tensor:
    def __init__(self, data):
        if isinstance(data, cp.ndarray):
            self.data = data
        else:
            self.data = cp.array(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    def __repr__(self):
        return f"Tensor({self.data}, shape={self.shape})"

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __sub__(self, other):  
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    def __truediv__(self, other):
        return Tensor(self.data / other.data)

    def __matmul__(self, other):
        return Tensor(self.data @ other.data)

    @property
    def T(self):
        return Tensor(self.data.T)

    def sum(self, axis=None):
        return Tensor(self.data.sum(axis=axis))

    def mean(self, axis=None):
        return Tensor(self.data.mean(axis=axis))

    def reshape(self, shape):
        return Tensor(self.data.reshape(shape))

    def exp(self):
        return Tensor(cp.exp(self.data))

    def log(self):
        return Tensor(cp.log(self.data))

    def max(self, axis=None):
        return Tensor(self.data.max(axis=axis))

    def relu(self):
        return Tensor(cp.maximum(self.data, 0))


a = Tensor([[1, 2], [3, 4]])
b = Tensor([[5, 6], [7, 8]])

print(a * b)           # element-wise
print(a.sum())         # 10
print(a.sum(axis=0))   # [4, 6]
print(a.sum(axis=1))   # [3, 7]
print(a.reshape((4,))) # [1, 2, 3, 4]