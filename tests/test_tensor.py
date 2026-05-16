import numpy as np
import cupy as cp
import pytest
import sys


sys.path.insert(0, '..')
from cybergrad.tensor import Tensor

def finite_diff(f, x, eps=1e-4):
    grad = np.zeros(x.data.shape)
    for idx in np.ndindex(x.data.shape):
        x.data[idx] += eps
        fp = float(f().data.sum().get())
        x.data[idx] -= 2*eps
        fm = float(f().data.sum().get())
        x.data[idx] += eps
        grad[idx] = (fp - fm) / (2*eps)
    return grad

def check(f, *tensors, eps=1e-4, tol=1e-3):
    out = f(*tensors)
    if not hasattr(out, 'backward'):
        out = out.sum() if out.data.ndim > 0 else out
    out.backward()
    for x in tensors:
        numeric = finite_diff(lambda: f(*tensors), x, eps)
        assert np.allclose(x.grad.get(), numeric, atol=tol), \
            f"grad mismatch\ngot:      {x.grad.get()}\nexpected: {numeric}"

def make():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    return a, b

def test_add():
    a, b = make()
    check(lambda a, b: (a + b).sum(), a, b)

def test_sub():
    a, b = make()
    check(lambda a, b: (a - b).sum(), a, b)

def test_mul():
    a, b = make()
    check(lambda a, b: (a * b).sum(), a, b)

def test_div():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[1.0, 2.0], [3.0, 4.0]])  # avoid div by zero
    check(lambda a, b: (a / b).sum(), a, b)

def test_matmul():
    a, b = make()
    check(lambda a, b: (a @ b).sum(), a, b)

def test_sum():
    a, _ = make()
    check(lambda a: a.sum(), a)

def test_mean():
    a, _ = make()
    check(lambda a: a.mean(), a)

def test_reshape():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    check(lambda a: a.reshape((4,)).sum(), a)

def test_exp():
    a = Tensor([[0.1, 0.2], [0.3, 0.4]])  
    check(lambda a: a.exp().sum(), a)

def test_log():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])  
    check(lambda a: a.log().sum(), a)

def test_relu():
    a = Tensor([[-1.0, 2.0], [-3.0, 4.0]])
    check(lambda a: a.relu().sum(), a)

def test_transpose():
    a, b = make()
    check(lambda a, b: (a.T @ b).sum(), a, b)

def test_chain():
    # tests multiple ops composed together
    a, b = make()
    check(lambda a, b: ((a @ b) + a).relu().sum(), a, b)