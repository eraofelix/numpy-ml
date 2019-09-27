from abc import ABC, abstractmethod
import numpy as np
from tests import assert_is_binary, assert_is_stochastic


class ObjectiveBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def grad(self, y_true, y_pred, **kwargs):
        pass


class SquaredError(ObjectiveBase):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def __str__(self):
        return "SquareError"

    @staticmethod
    def loss(y_true, y_pred, **kwargs):

        L = 0.5 * np.linalg.norm(y_pred - y) ** 2

        return L

    def grad(self, y_true, y_pred, z, act_fn):
        """Gradient of the squared error loss respect to the pre-nonlinearity
        input : z

        dL/dz = dL/pred * dpred/dz = (y_pred-y_true)*act_fn.grad(z)

        :param y_true:
        :param y_pred:
        :param z: input of activation layer
        :param act_fn: activation layer
        :return:
        """

        g = (y_pred-y_true)*act_fn.grad(z)

        return g


class CrossEntropy(ObjectiveBase):
    def __init__(self):
        super().__init__()

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def __str__(self):
        return "CrossEntropy"

    @staticmethod
    def loss(y_true, y_pred):
        """Cross-entropy (log) loss. Returns the sum (not average!) of the
        losses per-sample.

        :param y_true: (n, m) for n_samples and m_classes
        :param y_pred: (n, m)
        :return:
        """
        assert_is_binary(y_true)
        assert_is_stochastic(y_pred)
        eps = np.finfo(float).eps

        cross_entropy = -np.sum(y_true*np.log(y_pred+eps))

        return cross_entropy

    def grad(self, y_true, y_pred, **kwargs):
        """  ???????????????????????????????????????????????????
        Let:  f(z) = cross_entropy(softmax(z)).
        Then: df / dz = softmax(z) - y_true
                      = y_pred - y_true

        Note that this gradient goes through both the cross-entropy loss AND the
        softmax non-linearity to return df / dz (rather than df / d softmax(z) ).

        Input
        -----
        y : numpy array of shape (n, m)
            A one-hot encoding of the true class labels. Each row constitues a
            training example, and each column is a different class
        y_pred: numpy array of shape (n, m)
            The network predictions for the probability of each of m class labels on
            each of n examples in a batch.

        Returns
        -------
        grad : numpy array of shape (n, m)
            The gradient of the cross-entropy loss with respect to the *input*
            to the softmax function.
        """
        assert_is_binary(y_true)
        assert_is_stochastic(y_pred)
        g = y_pred - y_true

        return g
