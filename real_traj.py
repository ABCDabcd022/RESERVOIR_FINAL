import reservoir_predictor1 as res1
from nolitsa import data
import copy
import math
import numpy as np


def Norma (W):
    sigm = 0
    for i in range(len(W)):
        sigm = sigm + (math.fabs(W[i]))*2
    return math.sqrt(sigm)



# Опишем траектории руками
# ПЕРВАЯ ТРАЕКТОРИЯ
dt = 0.01
x0 = [0.62225717, -0.08232857, 30.60845379]

x0 = data.lorenz(length=100000, sample=dt, x0=x0,
               sigma=16.0, beta=4.0, rho=45.92)[1]

#11
time = copy.deepcopy(x0) #берем значения Лоренца
X0 = time[:,0] #берем только x
X0 = X0.reshape(100000,1) #зачем-то 
X0 = 2 * (X0 - X0.min()) / (X0.max() - X0.min()) - 1 #нормализация(?)

#model0 = res1.reservoir_predictor1(X=X0, prediction_steps = 15, exchange = 50000)
#pred, res, W = model0.do()
#w1 = Norma(W)

# ВТОРАЯ ТРАЕКТОРИЯ
dt = 0.01
x1 = [0.62225717 + 0.5, -0.08232857 + 0.5, 30.60845379 + 0.5]

x1 = data.lorenz(length=100000, sample=dt, x0=x1,
               sigma=16.0, beta=4.0, rho=45.92)[1]

#11
time = copy.deepcopy(x1) #берем значения Лоренца
X1 = time[:,0] #берем только x
X1 = X1.reshape(100000,1) #зачем-то 
X1 = 2 * (X1 - X1.min()) / (X1.max() - X1.min()) - 1 #нормализация(?)

#model1 = res1.reservoir_predictor1(X=X1)
#pred, res, W = model1.do()

# ТРЕТЬЯ ТРАЕКТОРИЯ
dt = 0.01
x2 = [0.62225717 - 0.5, -0.08232857 - 0.5, 30.60845379 - 0.5]

x2 = data.lorenz(length=100000, sample=dt, x0=x2,
               sigma=16.0, beta=4.0, rho=45.92)[1]

#11
time = copy.deepcopy(x2) #берем значения Лоренца
X2 = time[:,0] #берем только x
X2 = X2.reshape(100000,1) #зачем-то 
X2 = 2 * (X2 - X2.min()) / (X2.max() - X2.min()) - 1 #нормализация(?)

#model2 = res1.reservoir_predictor1(X=X2)
#pred, res, W = model2.do()

# ЧЕТВЁРТАЯ ТРАЕКТОРИЯ
dt = 0.01
x3 = [0.62225717 - 1.0, -0.08232857 - 1.0, 30.60845379 - 1.0]

x3 = data.lorenz(length=100000, sample=dt, x0=x3,
               sigma=16.0, beta=4.0, rho=45.92)[1]

#11
time = copy.deepcopy(x3) #берем значения Лоренца
X3 = time[:,0] #берем только x
X3 = X3.reshape(100000,1) #зачем-то 
X3 = 2 * (X3 - X3.min()) / (X3.max() - X3.min()) - 1 #нормализация(?)

#model3 = res1.reservoir_predictor1(X=X3)
#pred, res, W = model3.do()

X = []
for i in range(1):
    X.append(X0)
    X.append(X1)
    X.append(X2)
    X.append(X3)
X = np.concatenate((X0, X1, X2, X3), axis=1)

#model = res1.reservoir_predictor1(X=X)
#pred, res, W = model.do()

q = X[:10]

