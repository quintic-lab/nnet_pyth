from mlp.layer import *


class OutputLayer(Layer):

    def __init__(self, cost, activ, n_nodes, n_inputs, bias=True):
        super().__init__(activ, n_nodes, n_inputs, bias)
        self.costType = cost
        if not cost in ['sse', 'entropy']:
            print("Allowed error cost functions : 'sse','entropy'")
            #  costFunc is the error function (y-f)^2
        #  costFuncDeriv is derivative of the cost function w.r.t f
        #  for example 2(y-f)(-1)
        #  y is the target and f is the predicted value
        self.costFunc, self.costFuncDeriv = self.__costFunction()

    def __costFunction(self):
        if self.costType == 'sse':
            def tmp(y, f):
                return (y - f) ** 2

            def tmp1p(y, f):
                return 2 * (f - y)
        elif self.costType == 'entropy':
            def tmp(y, f):
                return -(y * np.log(f) + (1 - y) * np.log(1 - f))

            def tmp1p(y, f):
                return -(y / f - (1 - y) / (1 - f))
        else:
            pass
        return tmp, tmp1p

    def predict_1hid_layer(self, x, prevLayer):
        # default is one hidden layer
        o1 = prevLayer.feedData(x)
        o2 = self.feedData(o1)
        return o2

    def predict(self, x, Layer1, Layer2):
        # NNET with 3 layers (2 hidden) current is L3
        o1 = Layer1.feedData(x)
        o2 = Layer2.feedData(o1)
        o3 = self.feedData(o2)
        return o3

