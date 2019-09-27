from abc import ABC, abstractmethod
import numpy as np


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):

        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        raise NotImplementedError


class Sigmoid(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def fn(self, z):

        out = 1 / (1 + np.exp(-z))
        return out

    def grad(self, x, **kwargs):

        g = self.fn(x)*(1-self.fn(x))

        return g

    def grad2(self, x):

        g2 = self.grad(x) * (1 - self.fn(x)) - self.fn(x) * self.grad(x)

        return g2


class ReLU(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, z):

        out = np.clip(z, 0, np.inf)

        return out

    def grad(self, x, **kwargs):

        g = (x > 0).astype(int)

        return g

    def grad2(self, x):

        g2 = np.zeros_like(x)

        return g2

