import numpy as np


class Layer:

    def __init__(self, activ, n_nodes, n_inputs, bias=True):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.activ = activ
        self.biasVal = float(bias)  # this is to use later in the training
        self.acfunc, self.deriv_out = self.__generate_funcs()
        self.w = 2 * np.random.random([n_nodes, n_inputs + 1]) - 1
        if not activ in ['linear', 'sigmoid', 'tanh']:
            print("Allowed activation function : linear,sigmoid,tanh")
        # self.biasVa if no use of bias it is set to 0
        # the last value in the weight represent the bias
        if not bias:
            self.w[:, -1] = 0.0

    def feed(self, x):
        d = np.dot(self.w, np.hstack((x, 1.0)))
        return self.acfunc(d)

    def feedData(self, X):
        one = np.ones([X.shape[0], 1])
        d = np.dot(np.hstack((X, one)), self.w.T)
        return self.acfunc(d)

    def __generate_funcs(self):
        if self.activ == 'linear':
            def tmp(x):
                return x

            def tmp1p(y):
                return 1.0
        elif self.activ == 'sigmoid':
            def tmp(x):
                return 1.0 / (1.0 + np.exp(-x))

            def tmp1p(y):
                return y * (1 - y)
        elif self.activ == 'tanh':
            def tmp(x):
                return np.tanh(x)

            def tmp1p(y):
                return (1 - y * y)
        else:
            pass
        return tmp, tmp1p

    def bias(self):
        return self.w[:, -1]
