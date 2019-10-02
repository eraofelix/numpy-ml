from abc import ABC, abstractmethod
import numpy as np
from initializers import WeightInitializer, OptimizerInitializer, ActivationInitializer
from wrappers import init_wrappers, Dropout

from utils import (
    pad1D,
    pad2D,
    conv1D,
    conv2D,
    im2col,
    col2im,
    dilate,
    deconv2D_naive,
    calc_pad_dims_2D
)


class LayerBase(ABC):
    def __init__(self, optimizer=None):
        self.X = []  # 存储这一层的输入变量self.X.append(X)
        self.act_fn = None  # 激活函数
        self.trainable = True
        self.optimizer = OptimizerInitializer(optimizer)
        self.gradients = {}  # 梯度，维度与parameters一致
        self.parameters = {}  # 可学习参数
        self.hyperparameters = {}  # 超参
        self.derived_variables = {}  # 存储激活层前的值Y=act_fn(Z)中的Z

        super().__init__()

    @abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        raise NotImplementedError

    def freeze(self):
        self.trainable = False

    def unfreeze(self):
        self.trainable = True

    def flush_gradients(self):
        assert self.trainable, "Layer is frozen"
        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    def update(self, curr_loss=None):
        assert self.trainable, "Layer is frozen"
        self.optimizer.step()
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] = self.optimizer(self.parameters[k], v, k, curr_loss)
        self.flush_gradients()

    def set_params(self, summary_dict):
        layer, sd = self, summary_dict
        flatten_keys = ["parameters", "hyperparameters"]

        for k in flatten_keys:
            if k in summary_dict:
                entry = summary_dict[k]
                summary_dict.update(entry)
                del summary_dict[k]

        for k, v in summary_dict.items():
            if k in self.parameters:
                layer.parameters[k] = v
            if k in self.hyperparameters:
                if k == "act_fn":
                    layer.act_fn = ActivationInitializer(v)()
                if k == "optimizer":
                    layer.optimizer = OptimizerInitializer(summary_dict[k])()
                if k not in ["wrappers", "optimizer"]:
                    setattr(self, k, v)
                if k == "wrappers":
                    layer = init_wrappers(self, summary_dict[k])
        return layer

    def summary(self):
        return {
            "layer": self.hyperparameters["layer"],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters
        }


class FullyConnected(LayerBase):
    def __init__(self, n_out, act_fn=None, init="glorotuniform", optimizer=None):
        """ A fully-connected layer.
        Equations:

            Y = act_fn(W . X + b)

        :param n_out: The dimensionality of layer out
        :param act_fn: activation function
        :param init: weight initialization strategy
        :param act_fn: activation layer
        :param optimizer: the optimization method to use when performing
               gradient updates
        :return:
        """
        super().__init__(optimizer)

        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.act_fn = ActivationInitializer(act_fn)()
        self.parameters = {"W": None, "b": None}
        self.is_initialized = False

    def _init_params(self, **kwargs):
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
        b = np.zeros((1, self.n_out))
        W = init_weights(weight_shape=(self.n_in, self.n_out))

        self.parameters = {"W": W, "b": b}
        self.derived_variables = {"Z": []}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}

        self.is_initialized = True

    @property
    def hyperparameters(self):
        return {
            "layer": "FullyConnected",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            }
        }

    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y, Z = self._fwd(X)

        if retain_derived:
            self.X.append(X)
            self.derived_variables["Z"].append(Z)

        return Y

    def _fwd(self, X):
        """ 实际的计算函数

        :param X: The input

        :return: Y: Y = act_fn(W . X + b)
        :return: Z: Z = W . X + b
        """
        W = self.parameters["W"]
        b = self.parameters["b"]

        Z = X @ W + b
        Y = self.act_fn(Z)

        return Y, Z

    def backward(self, dLdY, retain_grads=True):
        """ 从层输出到层输入反向传播

        :param dLdY: loss关于层输出的梯度(n_ex, n_out),n_ex是前一层过来的维度(<--)
        :return: dLdX: loss关于层输入的梯度，再往前传就是前一层的dLdY！！！！！！

        """
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdY, list):
            dLdY = [dLdY]

        dLdX = []
        X = self.X
        for dLdy, x in zip(dLdY, X):
            dLdx, dLdw, dLdb = self._bwd(dLdy, x)
            dLdX.append(dLdx)
            if retain_grads:
                self.gradients["W"] += dLdw  # ??????
                self.gradients["b"] += dLdb

        return dLdX[0] if len(X) == 1 else dLdX

    def _bwd(self, dLdY, X):
        """ 计算loss关于X，W，b三者的梯度

        :param dLdY: loss关于层输出的梯度(n_ex, n_out),n_ex是前一层过来的维度(<--)
        :param X: 层输入
        :return: dLdX: loss关于层输入X的梯度
        :return: dLdW: loss关于层W的梯度
        :return: dLdB: loss关于层b的梯度
        """
        W = self.parameters["W"]
        b = self.parameters["b"]

        Z = X @ W + b  # Y = act_fn(Z)

        dLdZ = dLdY * self.act_fn.grad(Z)  # dL/dZ = dL/dY * dY/dZ

        dLdX = dLdZ * W.T  # dL/dX = dL/dZ * dZ/dX
        dLdW = X.T * dLdZ  # dL/dW = dL/dZ * dZ/dW
        dLdB = dLdZ.sum(axis=0, keep_dims=True)  # dL/dB = dL/dZ * dZ/dB

        return dLdX, dLdW, dLdB

    def _bwd2(self):
        raise NotImplementedError


class SoftMax(LayerBase):
    def __init__(self, dim=-1, optimizer=None):
        """ Softmax层

        Y = e^X / sum(e^X)

        :param dim: 计算softmax的维度
        :param optimizer: 优化器
        """
        super().__init__(optimizer)
        self.dim = dim
        self.n_in = None
        self.is_initialized = False

    def _init_params(self, **kwargs):
        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        return {
            "layer": "SoftmaxLayer",
            "n_in": self.n_in,
            "n_out": self.n_in,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, X, retain_derived=True):
        """ Softmax前向计算层

        Y = e^X / sum(e^X)

        :param X: shape(n_ex, n_in)， n_ex个n_in长度的输入
        :param retain_derived: Whether to retain the variables calculated during the forward pass
            for use later during backprop. If `False`
        """
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = self._fwd(X)

        if retain_derived:
            self.X.append(X)

    def _fwd(self, X):
        """Actual computation of softmax forward pass"""
        # center data to avoid overflow
        e_X = np.exp(X - np.max(X, axis=self.dim, keepdims=True))
        return e_X / e_X.sum(axis=self.dim, keepdims=True)

    def backward(self, dLdy):
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X

        for dy, x in zip(dLdy, X):
            dx = self._bwd(dy, x)
            dX.append(dx)

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """ 实际的反向计算

            我们设softmax输入x=[x1, x2,..., xn], 维度是1xn，
                 softmax输出y=[y1, y2,..., yn], 维度是1xn

            我们要计算的是dyi/dxj，也就是y中的每个元素yi关于x每个元素xj的导数，这是个Jacobian矩阵，
            维度是nxn

            其实很好推导，就是除法求导展开，求和展开更好理解，已推导
        """
        dX = []
        for dy, x in zip(dLdy, X):
            dxi = []
            for dyi, xi in zip(*np.atleast_2d(dy, x)):
                yi = self._fwd(xi.reshape(1, -1)).reshape(-1, 1)
                dyidxi = np.diagflat(yi) - yi @ yi.T
                dxi.append(dyi @ dyidxi)
            dX.append(dxi)
        return np.array(dX).reshape(*X.shape)
















