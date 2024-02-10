import reservoir_predictor1 as my

#Для построения архитектуры ESN использовалась библиотека reservoirPy, основанная на графовых вычислениях
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.datasets import to_forecasting # разбиение на train/test
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge #узлы резервуара и считывающего слоя
from reservoirpy.observables import nrmse, rsquare
import json
from scipy.integrate import solve_ivp
import copy
import warnings
from nolitsa import data

###ПОДГОТОВКА ДАННЫХ
#9
dt = 0.01
x0 = [0.62225717, -0.08232857, 30.60845379]


x = data.lorenz(length=10000, sample=dt, x0=x0,
               sigma=16.0, beta=4.0, rho=45.92)[1]

#11
time = copy.deepcopy(x) #берем значения Лоренца
X = time[:,0] #берем только x
X = X.reshape(10000,1) #зачем-то 
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1 #нормализация(?)

model = my.reservoir_predictor1(X=X, exchange=6000, prediction_steps = 100)
pred = model.do()