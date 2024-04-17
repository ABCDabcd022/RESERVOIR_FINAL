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

rpy.verbosity(0)
rpy.set_seed(42)  

X_train = X[:3000]
Y_train = X[1:3001]
##
reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

train_states = reservoir.run(X_train, reset=True)
reservoir = reservoir.reset()
ridge = ridge.fit(train_states, Y_train, warmup=1000)

esn_model = reservoir >> ridge

esn_model = esn_model.fit(X_train, Y_train, warmup=1000)
warmup_y = esn_model.run(X_train[:-1000], reset=True)

Y_pred = np.empty((6000, 3))
x = warmup_y[-1].reshape(1, -1)

for i in range(6000):
    x = esn_model(x)
    Y_pred[i] = x

def plot_dimension(dim, name):
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    ax.plot(time_steps[3000:], X[3000:][:, dim], color = "black") #реал.
    ax.plot(time_steps[3000:], Y_pred[3000:][:, dim], "--", color = "gray") #предск.
    plt.xlabel("time")
    plt.ylabel(name) 
    plt.draw()
    plt.show()

plot_dimension(0, 'x')
plot_dimension(1, 'y')
plot_dimension(2, 'z')

"""
def plot_dimension(dim, name):
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    ax.plot(time_steps, X[:, dim], color = "black") #реал.
    ax.plot(time_steps, Y_pred[:, dim], "--", color = "gray") #предск.
    plt.xlabel("time")
    plt.ylabel(name) 
    plt.draw()
    plt.show()

plot_dimension(0, 'x')
plot_dimension(1, 'y')
plot_dimension(2, 'z')
"""













