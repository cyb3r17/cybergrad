import cupy as cp

def _unbroadcast(grad, shape):
    ndim_diff = len(grad.shape) - len(shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    def __init__(self, data):
        if isinstance(data, cp.ndarray):
            self.data = data
        else:
            self.data = cp.array(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self):
        return f"Tensor({self.data}, shape={self.shape})"

    def __add__(self, other):
        out = Tensor(self.data + other.data)
        out._prev = {self, other}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            if other.grad is None: other.grad = cp.zeros_like(other.data)
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Tensor(self.data - other.data)
        out._prev = {self, other}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            if other.grad is None: other.grad = cp.zeros_like(other.data)
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad -= _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data)
        out._prev = {self, other}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            if other.grad is None: other.grad = cp.zeros_like(other.data)
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = Tensor(self.data / other.data)
        out._prev = {self, other}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            if other.grad is None: other.grad = cp.zeros_like(other.data)
            self.grad += out.grad / other.data  # fixed, no transpose
            other.grad += out.grad * (-self.data / other.data**2)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)  # fixed, @ not *
        out._prev = {self, other}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            if other.grad is None: other.grad = cp.zeros_like(other.data)
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    @property
    def T(self):
        out = Tensor(self.data.T)
        out._prev = {self}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            self.grad += out.grad.T  # transpose grad back
        out._backward = _backward
        return out

    def sum(self, axis=None,keepdims=False):
        out = Tensor(self.data.sum(axis=axis,keepdims=keepdims))
        out._prev = {self}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            self.grad += cp.ones_like(self.data) * out.grad  # fixed
        out._backward = _backward
        return out

    def mean(self, axis=None):
        out = Tensor(self.data.mean(axis=axis))
        out._prev = {self}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            self.grad += cp.ones_like(self.data) * out.grad / self.data.size
        out._backward = _backward
        return out

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape))
        out._prev = {self}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(cp.exp(self.data))
        out._prev = {self}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            self.grad += out.data * out.grad  # d/dx e^x = e^x = out.data
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(cp.log(self.data))
        out._prev = {self}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            self.grad += out.grad / self.data  # d/dx log(x) = 1/x
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(cp.maximum(self.data, 0))
        out._prev = {self}
        def _backward():
            if self.grad is None: self.grad = cp.zeros_like(self.data)
            self.grad += (self.data > 0) * out.grad  # 1 where input > 0, else 0
        out._backward = _backward
        return out

    def softmax(self, axis=1):
        e = self.exp()
        return e / e.sum(axis=axis, keepdims=True)

    def zero_grad(self):
        self.grad = None

    def backward(self):
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        build_topo(self)
        self.grad = cp.ones_like(self.data)
        for node in reversed(topo):
            node._backward()



