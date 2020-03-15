"""
This is an mlp class with a maximum of 2 hidden layers and the bias is set to 0
- further version will implement the bias

Kamel Saadi 14/03/2020
repo: https://github.com/quintic-lab

"""


import numpy as np

obj_func = {
    'sse': [lambda y, f: float(sum((y - f)**2)),
            lambda y, f: -2*(y-f) ],
    'entropy': [lambda y, f: -(y * np.log(f) + (1 - y) * np.log(1 - f)),
                lambda y, f: -(y / f - (1 - y) / (1 - f)) ]
}


class mlp:

    col = (-1,1)
    row = (1,-1)

    def __init__(self, learning_rate, n_epochs, batch_size=-1,
                 error_func='sse', shuffle=False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_epochs = n_epochs
        if error_func.lower() not in ['sse', 'entropy']:
            raise ValueError("error_func accepts only two value 'sse' and 'entropy")
        self.cost_func       = obj_func[error_func.lower()][0]
        self.cost_func_deriv = obj_func[error_func.lower()][1]

        self.n_layers = 0
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)
        self.n_layers += 1


    def train(self, x_train, y_train, x_valid, y_valid):
        layers = self.layers
        eta = self.learning_rate
        for epoch in range(self.n_epochs):
            for obs in range(x_train.shape[0]):
                o0, y = x_train[obs, :], float(y_train[obs])
                o1 = layers[0].feed(o0)
                o2 = layers[1].feed(o1)
                o3 = float(layers[2].feed(o2))

                dCost_do3 = self.cost_func_deriv(y, o3) # 2 * (o3 - y)

                delta3 = dCost_do3 * layers[2].deriv_out(o3)
                layers[2].w += -eta * delta3 * o2.reshape(self.row)

                delta2 = delta3 * layers[1].deriv_out(o2)
                layers[1].w += -eta * delta2 * o1.reshape(self.row)

                delta1 = np.dot(layers[0].deriv_out(o1), delta2.T)
                layers[0].w = -eta * np.sum( delta1 * o0.reshape(1, -1),
                                             axis=1).reshape(-1, 1)

            train_err = self.cost_func( y_train, self.predict(x_train) )
            valid_err = self.cost_func( y_valid, self.predict(x_valid) )
            print("epoch= %4d   train_error =  %6.4f   test_error =  "
                  "%6.4f"%(epoch, train_err, valid_err))

    def predict(self, x):
        pred = x.copy()
        for layer in self.layers:
            pred = layer.feedData(pred)
        return pred






