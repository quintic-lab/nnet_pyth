# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:20:54 2017

@author: ks_work

This works tested as of 12/03/2020

2 hidden layers
"""
import numpy as np
from mlp.layer import Layer
from mlp.output_layer import OutputLayer


# class layer:
#     def __init__(self, activ, n_nodes, n_inputs):
#         self.n_nodes = n_nodes
#         self.n_inputs = n_inputs
#         self.activ = activ
#         self.acfunc, self.deriv_out = self.__generate_funcs()
#         self.w = 2 * np.random.random([n_nodes, n_inputs]) - 1
#         if not activ in ['linear', 'sigmoid', 'tanh']:
#             print("Allowed activation function : linear,sigmoid,tanh")
#
#     def feed(self, x):
#         return self.acfunc(np.dot(self.w, x.reshape(-1, 1)))
#
#     def feedData(self, X):
#         d = np.dot(X, self.w.T)
#         return self.acfunc(d)
#
#     def __generate_funcs(self):
#         if self.activ == 'linear':
#             def tmp(x):
#                 return x
#
#             def tmp1p(y):
#                 return 1.0
#         elif self.activ == 'sigmoid':
#             def tmp(x):
#                 return 1.0 / (1.0 + np.exp(-x))
#
#             def tmp1p(y):
#                 return y * (1 - y)
#         elif self.activ == 'tanh':
#             def tmp(x):
#                 return np.tanh(x)
#
#             def tmp1p(y):
#                 return (1 - y * y)
#         else:
#             pass
#         return tmp, tmp1p
#
    # --------------------------------------------------------------------------


X = np.linspace(-10, 10, 500).reshape(-1, 1)
n, m = X.shape
ytrue = np.sin(X) / X

Y = ytrue + 0.4 * np.random.randn(n, m)

n_te = int(n * 0.5)
import random

te_ind = random.sample(range(n), n_te)
tr_ind = ~np.isin(range(n), te_ind)

Xte, Yte = X[te_ind, :], Y[te_ind]
Xtr, Ytr = X[tr_ind, :], Y[tr_ind]





in_dim = Xte.shape[1]

np.random.seed(3145)
# n_nodes = 250
# L1 = Layer('tanh', n_nodes, in_dim)
# L2 = Layer('tanh', 50, n_nodes)
# L3 = Layer('tanh', 1, 50)


n_nodes = 3
L1 = Layer('tanh', n_nodes, in_dim)
L2 = Layer('tanh', 2, n_nodes)
L3 = OutputLayer('sse', 'tanh', 1, 2 )  #




eta = 0.001
iters, sErr = 2000, np.Infinity
for i in range(iters):
    for obs in range(Xtr.shape[0]):
        o0, y = Xtr[obs, :], float(Ytr[obs])
        o1 = L1.feed(o0)
        o2 = L2.feed(o1)
        o3 = float(L3.feed(o0))

        #dCost_do3 = 2*(o3-y)
        dCost_do3 = L3.costFuncDeriv(y, o3)

        delta3 = dCost_do3 * L3.deriv_out(o3)
        L3.w += -eta * delta3 * np.hstack((o2, L3.biasVal))


        delta2 = delta3 * L2.deriv_out(o2)
        L2.w += -eta * delta2 *np.hstack((o1, L2.biasVal))

        delta1 = np.dot(L1.deriv_out(o1), delta2.T)
        for node_id in range(L2.n_nodes):
            L1.w += -eta * delta1[:, node_id].reshape(-1, 1) * o0

    _tr = L3.costFunc(Ytr, L3.score(Xtr, L1))
    _te = L3.costFunc(Yte, L3.score(Xte, L1))

    train_err = float(sum(_tr))
    test_err = float(sum(_te))
    sErr0 = sErr
    sErr = test_err
    if (sErr - sErr0) / sErr0 > 1e-6 and i > 50:
        break
    print("epoch = ", i, " Validation =", round(sErr, 4), " Training =",
          round(train_err, 4))
# import  matplotlib.pyplot as plt
#
# pred  = score(X)
# plt.figure(figsize=(7,6))
# plt.plot(X,Y,'.', label='noise sine')
# plt.plot(X,ytrue,'-k', label='true sine')
# plt.plot(X,pred,'-r', label='nnet')
# plt.grid()
# plt.xlabel('x')
# plt.legend(loc='upper right')
# plt.title('NNET regression')
# plt.show()











