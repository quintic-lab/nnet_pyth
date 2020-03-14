import numpy as np

activ_func = {
    'identity':[ lambda x: x, lambda x:1 ],
    'sigmoid':[lambda x: 1.0/(1.0+np.exp(-x)) , lambda x: x*(1-x) ],
    'tanh'   :[lambda x: np.tanh(x) , lambda x: 1-x*x]
}


class LayerNoBias:
    def __init__(self, activ, n_nodes, n_inputs):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.activ = activ
        self.w = 2 * np.random.random([n_nodes, n_inputs]) - 1
        if not activ.lower() in ['linear', 'sigmoid', 'tanh']:
            print("Allowed activation function : linear,sigmoid,tanh")
        self.acfunc, self.deriv_out = activ_func[activ.lower()]

    def feed(self, x):
        return self.acfunc(np.dot(self.w, x.reshape(-1, 1)))

    def feedData(self, X):
        d = np.dot(X, self.w.T)
        return self.acfunc(d)