import reservoirpy as rpy

rpy.verbosity(0)  # no need to be too verbose here
rpy.set_seed(42)  # make everyhting reproducible !

from reservoirpy.nodes import Reservoir, Ridge
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import reservoirpy as rpy

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
dt = 0.01

def f(state, t):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

x_train_length = 60.0 

state0 = [1.0, 1.0, 1.0]
time_steps = np.arange(0.0, x_train_length, dt) 

X = odeint(f, state0, time_steps)
Xx = np.empty((6000, 1)) 

for i in range(6000):
    Xx[i] = X[i][0]

X = Xx

rpy.verbosity(0)
rpy.set_seed(42)  

X_train = X[:30]
Y_train = X[1:31]

##Генерация и долгосрочное прогнозирование

reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge

esn_model = esn_model.fit(X_train, Y_train, warmup=20)
warmup_y = esn_model.run(X_train[:-20], reset=True)

Y_pred = np.empty((60, 1))
x = warmup_y[-1].reshape(1, -1)

for i in range(60):
    x = esn_model(x)
    Y_pred[i] = x

Xx = np.empty((60, 1))
for i in range(60):
    Xx[i] = X[i]

X = Xx

def plot_dimension():
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    ax.plot(X, color = "blue") #реал.
    ax.plot(Y_pred, "--", color = "gray") #предск.
    plt.xlabel("time")
    plt.draw()
    plt.show()

plot_dimension()





