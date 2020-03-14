"""
- Artificial Neural Network with 1 hidden Layer
- Cost Function : SSE
- Activation Function : tanh

- Data : generated noisy sinc function


Author : Dr Kamel Saadi
Date   : 12/12/2017

incorporating bias for each node
most likely to enable bias only in the output node.

As we have seen in previous examples, if we don't use bias term
the sinc function nnet learning won't work

This runs fine 14/03/2020
"""

import numpy as np
from mlp.layer import Layer
from mlp.output_layer import OutputLayer


#----------------------------------------------------------
np.random.seed(345)
XI = np.linspace(-10,10,500).reshape(-1,1)
n = XI.shape[0]
ytrue = np.sin(XI)/XI
YI = ytrue + 0.2*np.random.randn(n).reshape(n,1)


import random

n_tr = int(0.5*n)  # split 50/50

tr_ix = random.sample(range(n),n_tr)
te_ix = ~np.isin(range(n), tr_ix)


Xtr, Ytr = XI[tr_ix,:], YI[tr_ix,:]
Xte, Yte = XI[te_ix,:], YI[te_ix,:]

#------------------------------------------------------------------

np.random.seed(3148)
# np.random.seed(45)
# n_nodes = 30
in_dim = XI.shape[1]

# --------------------------------------------------------------------------

# np.random.seed(3148) good combo
# n_nodes = 30
# L1    = layer('tanh',n_nodes ,in_dim, True) # no bias
# L2    = output_layer('sse','tanh',1, n_nodes, True) # bias


n_nodes = 60
L1 = Layer('tanh', n_nodes, in_dim)

L2 = OutputLayer('sse', 'identity', 1, n_nodes  )  #

eta = 0.001

iters, sErr = 50, np.Infinity
TrEnt = []
TeEnt = []
for i in range(iters):
    # np.random.shuffle(X)
    for obs in range(Xtr.shape[0]):
        o0, y = Xtr[obs, :], float(Ytr[obs])
        o1 = L1.feed(o0)
        o2 = float(L2.feed(o1))

        dCost_do2 = L2.costFuncDeriv(y, o2)

        delta2 = dCost_do2 * L2.deriv_out(o2)
        # here eta*delta applies to the last column of
        # the resulting hstack, i.e. the bias
        L2.w += -eta * delta2 * np.hstack((o1, L2.biasVal))

        delta1 = delta2 * L1.deriv_out(o1)
        L1.w += -eta * delta1.reshape(-1, 1) * np.hstack((o0, L1.biasVal))

    _tr = L2.costFunc(Ytr, L2.predict_1hid_layer(Xtr, L1))
    _te = L2.costFunc(Yte, L2.predict_1hid_layer(Xte, L1))

    train_err = float(sum(_tr))
    test_err = float(sum(_te))
    sErr0 = sErr
    sErr = test_err
    if (sErr - sErr0) / sErr0 > 1e-6 and i > 50:
        break
    print("epoch = ", i, " Validation =", round(sErr, 4), " Training =",
          round(train_err, 4))
    TrEnt.append(train_err)
    TeEnt.append(test_err)


















