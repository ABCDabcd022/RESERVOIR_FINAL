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

class reservoir_predictor1:
    def __init__(self, 
                X, 
                y_pred = [],
                prediction_steps = 10,
                exchange = 2000,
                units = 100,
                leak_rate = 0.3,
                spectral_radius = 1.25,
                input_scaling = 1.0,
                connectivity = 0.1,
                input_connectivity = 0.2,
                regularization = 1e-8,
                seed = 1234):
        self.X = X
        x, y = to_forecasting(X, forecast=prediction_steps)
        X_train1, y_train1 = x[:exchange], y[:exchange]
        X_test1, y_test1 = x[exchange:], y[exchange:]
        
        self.X_train = X_train1
        self.y_train = y_train1
        self.X_test = X_test1
        self.y_test = y_test1
        
        self.y_pred = y_pred
        
        self.prediction_steps = prediction_steps
        self.exchange = exchange

        self.units = units
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.connectivity = connectivity
        self.input_connectivity = input_connectivity
        self.regularization = regularization
        self.seed = seed
    
    def train_test(self):
        sample = 2000
        test_len = self.X_test.shape[0]
        fig = plt.figure(figsize=(15, 5))
        plt.plot(np.arange(0, 2000), self.X_train[-sample:], label="X_train")
        plt.plot(np.arange(0, 2000), self.y_train[-sample:], label="y_train")
        plt.plot(np.arange(2000, 2000+test_len), self.X_test, label="X_test")
        plt.plot(np.arange(2000, 2000+test_len), self.y_test, label="y_test")
        plt.xlabel(r'$t$')
        plt.legend()
        plt.show()
        
    def results(self, sample=1500):
        fig = plt.figure(figsize=(15, 7))
        ax = plt.subplot(211)
        ax.plot(np.arange(sample), self.y_test[len(self.y_test)-sample:], lw=2, label="True value", color="black")
        ax.plot(np.arange(sample), self.y_pred[len(self.y_pred)-sample:], lw=3, label="ESN prediction", color="gray", linestyle="--")
        #ax.axvline(x=sample-len(self.X_test), color="blue")  # Add vertical line
        ax.axvline(x=sample-self.prediction_steps, color="red")  # Add vertical line
        ax.legend()
        plt.show()

    def do(self):
        ###HERE
        self.train_test()

        #узел резервуара
        reservoir = Reservoir(self.units, input_scaling=self.input_scaling, sr=self.spectral_radius,
                            lr=self.leak_rate, rc_connectivity=self.connectivity,
                            input_connectivity=self.input_connectivity, seed=self.seed)
        #выходной слой(считывающее устройство)
        readout = Ridge(1, ridge=self.regularization) # 1 - кол-во выходных нейронов
        #соединяем узлы и таким образом получаем модель 
        esn = reservoir >> readout

        #обучение модели
        esn = esn.fit(self.X_train, self.y_train)
        #делаем предсказание
        self.y_pred = esn.run(self.X_test)
        self.results()
        ###HERE1
        return self.y_pred

###ПОДГОТОВКА ДАННЫХ
#9
dt = 0.01
x0 = [0.62225717, -0.08232857, 30.60845379]


x = data.lorenz(length=4000, sample=dt, x0=x0,
               sigma=16.0, beta=4.0, rho=45.92)[1]

#11
time = copy.deepcopy(x) #берем значения Лоренца
X = time[:,0] #берем только x
X = X.reshape(4000,1) #зачем-то 
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1 #нормализация(?)

model = reservoir_predictor1(X=X)
pred = model.do()

model = reservoir_predictor1(X=X, prediction_steps = 100)
pred = model.do()
