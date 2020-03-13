import numpy as np

activ_func = {
    'identity':[ lambda x: x, lambda x:1 ],
    'sigmoid':[lambda x: 1.0/(1.0+np.exp(-x)) , lambda x: x*(1-x) ],
    'tanh'   :[lambda x: np.tanh(x) , lambda x: 1-x*x]
}

class Layer:

    def __init__(self, activ, n_nodes, n_inputs, bias=True):

        self.n_nodes  = n_nodes
        self.n_inputs = n_inputs
        self.activ = activ
        self.biasVal = float(bias)  # this is to use later in the training
        self.acfunc, self.deriv_out = activ_func[activ]

        self.w = 2 * np.random.random([n_nodes, n_inputs + 1]) - 1
        if not activ in ['identity', 'sigmoid', 'tanh']:
            print("Allowed activation function : linear,sigmoid,tanh")

        if not bias:
            self.w[:, -1] = 0.0

    def feed(self, x):
        d = np.dot(self.w, np.hstack((x, 1.0)))
        return self.acfunc(d)

    def feedData(self, X):
        one = np.ones([X.shape[0], 1])
        d = np.dot(np.hstack((X, one)), self.w.T)
        return self.acfunc(d)

    def bias(self):
        return self.w[:, -1]
