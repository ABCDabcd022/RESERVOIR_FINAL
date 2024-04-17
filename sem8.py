import reservoir_predictor1 as res1
from nolitsa import data
import copy

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
pred, res = model.do()

#print(res)

#def hyperparametric_optimization(step=1.0, qt=50):
    #reservoirs = [res1.reservoir_predictor1 for i in range(20)]
    #return reservoirs

#reservoirs = hyperparametric_optimization()
#res = reservoirs[5]().do

#print(res)
