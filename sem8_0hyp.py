import reservoir_predictor1 as res1
from nolitsa import data
import copy
import math

dt = 0.01
x0 = [0.62225717, -0.08232857, 30.60845379]


x = data.lorenz(length=100000, sample=dt, x0=x0,
               sigma=16.0, beta=4.0, rho=45.92)[1]

#11
time = copy.deepcopy(x) #берем значения Лоренца
X = time[:,0] #берем только x
X = X.reshape(100000,1) #зачем-то 
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1 #нормализация(?)

model = res1.reservoir_predictor1(X=X)
#pred, res = model.do()
#model = res1.reservoir_predictor1(X=X, input_connectivity = 0)
#pred, res = model.do()

pred, res, W = model.do()
print("ТУТ")
print(W)
print(type(W))
print(W[0])

#model.hyperparametric_optimization()

def Norma (W):
    sigm = 0
    for i in range(len(W)):
        sigm = sigm + (math.fabs(W[i]))*2
    return math.sqrt(sigm)

print(Norma(W))
    