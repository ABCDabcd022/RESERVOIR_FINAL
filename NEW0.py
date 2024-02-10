#Для построения архитектуры ESN использовалась библиотека reservoirPy, основанная на графовых вычислениях
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.datasets import to_forecasting # разбиение на train/test
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge #узлы резервуара и считывающего слоя
from reservoirpy.observables import nrmse, rsquare
import json
from scipy.integrate import solve_ivp

"""
#Определяется функция lorenz, которая принимает несколько параметров:
def lorenz(
    n_timesteps: int, #  количество временных шагов, на которые будет производиться интегрирование.
    #параметры системы Лоренца
    rho: float = 28.0, 
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    x0: [list, np.ndarray] = [1.0, 1.0, 1.0], # начальное состояние системы в виде списка или массива NumPy
    h: float = 0.03, # шаг интегрирования
    **kwargs, # дополнительные аргументы, которые могут быть переданы функции solve_ivp
) -> np.ndarray:
   
 #Функция описывающая систему Лоренца принимает время t и текущее состояние системы state (x, y, z) 
 #и возвращает производные по времени для каждой переменной.
    def lorenz_diff(t, state):
        x, y, z = state
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    t_eval = np.arange(0.0, n_timesteps * h, h) #массив t_evalсодержит значения времени для оценки решения на каждом шаге интегрирования
# Численной интегрирование системы  
#В качестве ргументов функцию lorenz_diff, начальное состояние x0, интервал времени (0.0, n_timesteps * h) 
#и массив t_eval, а также дополнительные аргументы kwargs, переданные в функцию lorenz.
    sol = solve_ivp(
        lorenz_diff, y0=x0, t_span=(0.0, n_timesteps * h), t_eval=t_eval, **kwargs
    )
# Возвращается транспонированное решение системы, размерность (n_timesteps, 3)

    return sol.y.T
"""

"""
#1
finish_mas = lorenz(n_timesteps=4000,
    rho=0.5,
    sigma=10.0,
    beta= 8/3,
    x0=[0.1, 1., 1.])

plt.figure()
plt.xlabel("$t$")
plt.ylabel("$x y z$")
plt.title("Lorenz timeseries")
plt.plot(finish_mas[:1000,0], color='blue')
plt.plot(finish_mas[:1000,1], color='red')
plt.plot(finish_mas[:1000,2], color='green')
plt.show()



#2
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*finish_mas.T, lw=0.5)
ax.set_xlabel("X ")
ax.set_ylabel("Y ")
ax.set_zlabel("Z  ")
ax.set_title("Lorenz Attractor")

plt.show()



#3
finish_mas1 = lorenz(n_timesteps=4000,
    rho=10,
    sigma=10.0,
    beta= 8/3,
    x0=[0.1, 1., 1.])

plt.figure()
plt.xlabel("$t$")
plt.ylabel("$x y z$")
plt.title("Lorenz timeseries")
plt.plot(finish_mas1[:1000,0], color='blue')
plt.plot(finish_mas1[:1000,1], color='red')
plt.plot(finish_mas1[:1000,2], color='green')
plt.show()



#4
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*finish_mas1.T, lw=0.5, color='red')
ax.set_xlabel("X ")
ax.set_ylabel("Y ")
ax.set_zlabel("Z  ")
ax.set_title("Lorenz Attractor")

plt.show()
"""

"""
#5
finish_mas2 = lorenz(n_timesteps=4000,
    rho=15,
    sigma=10.0,
    beta= 8/3,
    x0=[0.1, 1., 1.])

plt.figure()
plt.xlabel("$t$")
plt.ylabel("$x y z$")
plt.title("Lorenz timeseries")
plt.plot(finish_mas2[:1000,0], color='blue')
plt.plot(finish_mas2[:1000,1], color='red')
plt.plot(finish_mas2[:1000,2], color='green')
plt.show()

#6
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*finish_mas2.T, lw=0.5, color='green')
ax.set_xlabel("X ")
ax.set_ylabel("Y ")
ax.set_zlabel("Z  ")
ax.set_title("Lorenz Attractor")

plt.show()

#7
finish_mas4 = lorenz(n_timesteps=4000,
    rho=28,
    sigma=10.0,
    beta= 8/3,
    x0=[0., 1., 1.05])

plt.xlabel("$t$")
plt.ylabel("$x y z$")
plt.title("Lorenz timeseries")
plt.plot(finish_mas4[:1000,0], color='blue')
plt.plot(finish_mas4[:1000,1], color='red')
plt.plot(finish_mas4[:1000,2], color='green')
plt.show()


#8
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*finish_mas4.T, lw=0.3, color='green')
ax.set_xlabel("X ")
ax.set_ylabel("Y ")
ax.set_zlabel("Z  ")
ax.set_title("Lorenz Attractor")

plt.show()
"""

###ПОДГОТОВКА ДАННЫХ
#9
import warnings
from nolitsa import data
dt = 0.01
x0 = [0.62225717, -0.08232857, 30.60845379]


x = data.lorenz(length=4000, sample=dt, x0=x0,
               sigma=16.0, beta=4.0, rho=45.92)[1]
"""
plt.plot(range(len(x)),x)
plt.show()

plt.xlabel("$t$")
plt.ylabel("$x y z$")
plt.title("Lorenz timeseries")
plt.plot(x[:1000,0], color='blue')
plt.plot(x[:1000,1], color='red')
plt.plot(x[:1000,2], color='green')
plt.show()
"""

"""
#10
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*x.T, lw=0.5, color='green')
ax.set_xlabel("X ")
ax.set_ylabel("Y ")
ax.set_zlabel("Z  ")
ax.set_title("Lorenz Attractor")

plt.show()
"""

#11
import copy
time = copy.deepcopy(x) #берем значения Лоренца
X = time[:,0] #берем только x
X = X.reshape(4000,1) #зачем-то 
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1 #нормализация(?)

def train_test(X_train, y_train, X_test, y_test):
    sample = 2000
    test_len = X_test.shape[0]
    fig = plt.figure(figsize=(15, 5))
    plt.plot(np.arange(0, 2000), X_train[-sample:], label="X_train")
    plt.plot(np.arange(0, 2000), y_train[-sample:], label="y_train")
    plt.plot(np.arange(2000, 2000+test_len), X_test, label="X_test")
    plt.plot(np.arange(2000, 2000+test_len), y_test, label="y_test")
    plt.xlabel(r'$t$')
    plt.legend()
    plt.show()
    
def results(y_pred, y_test, sample=1500):

    fig = plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")
    plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
    

    plt.legend()
    plt.show()
###HERE0
from reservoirpy.datasets import to_forecasting

x, y = to_forecasting(X, forecast=10)

X_train1, y_train1 = x[:2000], y[:2000]
X_test1, y_test1 = x[2000:], y[2000:]

train_test(X_train1, y_train1, X_test1, y_test1)

#Первый тестовый набор параметров 
units = 100
leak_rate = 0.3
spectral_radius = 1.25
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.2
regularization = 1e-8
seed = 1234

#узел резервуара
reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)
#выходной слой(считывающее устройство)
readout   = Ridge(1, ridge=regularization) # 1 - кол-во выходных нейронов
#соединяем узлы и таким образом получаем модель 
esn = reservoir >> readout

#обучение модели
esn = esn.fit(X_train1, y_train1)
#делаем предсказание
y_pred1 = esn.run(X_test1)
results(y_pred1, y_test1, sample=1500)
###HERE1
"""
###НАСТРОЙКА СЕТИ
#12
#Первый тестовый набор параметров 
units = 100
leak_rate = 0.3
spectral_radius = 1.25
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.2
regularization = 1e-8
seed = 1234
"""

"""
#узел резервуара
reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)
#выходной слой(считывающее устройство)
readout   = Ridge(1, ridge=regularization) # 1 - кол-во выходных нейронов
#соединяем узлы и таким образом получаем модель 
esn = reservoir >> readout

###ОБУЧЕНИЕ
#обучение модели
esn = esn.fit(X_train1, y_train1)
#делаем предсказание, ЗДЕСЬ МЫ ПОКА ПРЕДСКАЗЫВАЕМ ТО, ЧТО И ТАК ЗНАЕМ
y_pred1 = esn.run(X_test1)
results(y_pred1, y_test1, sample=1500)

"""
"""
#13
from reservoirpy.nodes import FORCE

reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)
#  FORCE, который позволяет выполнять обучение модели онлайн, то есть постепенно, по мере получения новых данных,
#в отличие от оффлайн-обучения, когда все данные доступны заранее.
readout   = FORCE(1)


esn_online = reservoir >> readout

esn_online.train(X_train1, y_train1)
pred_online = esn_online.run(X_test1)
results(pred_online, y_test1, sample=1500)

#14
###ПРЕДСКАЗАНИЕ ПЕРВЫХ 15ти НЕИЗВЕСТНЫХ ШАГОВ

#что это?
#  функция, которая выполняет прогоны моделей с разными наборами параметров и вычисляет метрики оценки этих моделей 
def objective(dataset, config, *, iss, N, sr, lr, ridge, seed):
    
    
#config - словарь с настройками и гиперпараметрами модели.

    train_data, validation_data = dataset 
    X_train, y_train = train_data
    X_val, y_val = validation_data

    
    instances = config["instances_per_trial"] #задает количество прогонов модели для каждого набора параметров
    
    
    variable_seed = seed # свое начальное значение для каждого экземпляра
    
    losses = []; r2s = []; # списки для наших метрик
    
    for n in range(instances):
        #собираем модельку с определенными параметрами
        reservoir = Reservoir(N, 
                              sr=sr, 
                              lr=lr, 
                              inut_scaling=iss, 
                              seed=variable_seed)
        
        readout = Ridge(ridge=ridge)

        model = reservoir >> readout


        #Тренировка / тест
        predictions = model.fit(X_train, y_train) \
                           .run(X_test)
        
        loss = nrmse(y_test, predictions, norm_value=np.ptp(X_train))
        r2 = rsquare(y_test, predictions)
        
        # тут мы меняем начальное значение для инициализации резервуара каждый раз на новой модели
        variable_seed += 1
        # закидываем ошибки и метрики в списки
        losses.append(loss)
        r2s.append(r2)

    
    # Возврат словаря с метриками
    return {'loss': np.mean(losses),
            'r2': np.mean(r2s)}



#что это?
hyperopt_config = {
    "exp": f"hyperopt-multiscroll", # название эксперимента
    "hp_max_evals": 200, # кол-во наборов параметров, которые должен попробовать  hyperopt
    "hp_method": "random", # тут мы можем определить какой метод использовать для выбора этих наборов - случайный    
    "seed": 42, # начальное значение случайного состояния                     
    "instances_per_trial": 3, # кол-во моделей, которое будет опробовано с каждым набором параметров  
    #исследуемые диапазоны параметров
    "hp_space": {          
        #"choice" - фиксированное значение
        #"loguniform" - равномерное логарифмическое распределение на каком - то интервале
        
        "N": ["choice", 500], # количество нейронов в резервуаре            
        "sr": ["loguniform", 1e-2, 10], # спектральный радиус  
        "lr": ["loguniform", 1e-3, 1],  # скорость утечки
        "iss": ["choice", 0.9], # масшиабирование входных данных         
        "ridge": ["choice", 1e-7], # параметр регуляризации      
        "seed": ["choice", 1234]  # случайное начальное значение для инициализации резервуара   
}}



#что это?
with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)


from reservoirpy.hyper import research
train_len = 1000

X_train = X[:train_len]
y_train = X[15 : train_len + 15]

X_test = X[train_len : -15]
y_test = X[train_len + 15:]
#Вытаскиваем выборку, которая будет датасетом для нашего research, с помощью которого мы найдем лучшие параметры,
#дающие наименьшую ошибку
dataset = ((X_train, y_train), (X_test, y_test))

best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")
"""





