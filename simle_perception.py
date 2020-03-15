import numpy as np
#import matplotlib.pyplot as plt
import random
np.random.seed(123)
x = np.linspace(-5,5,200)

y = 2.5*x + 1 + 2*np.random.randn(len(x))

#plt.plot(x,y,'.')

np.random.seed(123)
w, b = np.random.rand(2)
epochs = 30
eta = 0.001
for e in range(epochs):
    mse = 0
    for xi, yi in zip(x, y):
        pred = w*xi + b
        dy = pred-yi
        w = w - eta*dy*xi
        b = b - eta*dy
        mse += dy**2
    print("w = %6.4f  b = %6.4f  mse = %6.4f"%(w,b,mse) )


#==================================================================
#np.random.seed(123)
random.seed(123)
x = np.linspace(-5,5,200)
y = 2.5*x + 1 + 2*np.random.randn(len(x))
# using batches

w, b = np.random.rand(2)
epochs = 20
eta = 0.001
shuffle = True
n = len(y)
batch_size = 15
n_batches = n//batch_size

random.seed(111)
for e in range(epochs):
    mse = 0
    if shuffle:
        index = random.sample(range(n),n )
        x = x[index]
        y = y[index]
    for i in range(n_batches):
        start = i*batch_size
        end = min(start + batch_size, n-1)
        xt = x[start:end]
        yt = y[start:end]
        pred = w*xt + b
        dy = pred - yt
        w = w - eta * np.dot(dy,xt)
        b = b - eta * sum(dy)
        mse += np.dot(dy,dy)
    print("w = %6.4f  b = %6.4f  mse = %6.4f" % (w, b, mse))



