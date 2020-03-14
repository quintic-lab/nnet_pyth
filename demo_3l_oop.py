
"""
14/03/2020

2 hidden layers but no bias

Modelling Sinc function with neural net
runs fine
OOP Code redesign works as well

- Prediction is not good, this work is to make sure initial implementation works
"""
import random
import numpy as np
from mlp.layer_no_bias import LayerNoBias
from mlp.mlp import mlp

# fixing random set for reproducibility
random.seed(123)
np.random.seed(3145)
# Generating Sinc function data
X = np.linspace(-10, 10, 500).reshape(-1, 1)
n, m = X.shape
ytrue = np.sin(X) / X
# adding noise
Y = ytrue + 0.4 * np.random.randn(n, m)

# train and test split
n_te = int(n * 0.5)
te_ind = random.sample(range(n), n_te)
tr_ind = ~np.isin(range(n), te_ind)
Xte, Yte = X[te_ind, :], Y[te_ind]
Xtr, Ytr = X[tr_ind, :], Y[tr_ind]

# definint the network
nnet = mlp(learning_rate = 0.001, n_epochs=100)
in_dim = Xte.shape[1]
n_nodes = 250
L1 = LayerNoBias('tanh', n_nodes, in_dim)
L2 = LayerNoBias('tanh', 50, n_nodes)
L3 = LayerNoBias('identity', 1, 50)
nnet.add_layer(L1)
nnet.add_layer(L2)
nnet.add_layer(L3)

#training the network
nnet.train(Xtr, Ytr, Xte, Yte)













