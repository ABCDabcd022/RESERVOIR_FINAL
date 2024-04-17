##Учебная программа
##1
import reservoirpy as rpy

rpy.verbosity(0)
rpy.set_seed(42)  # сделать все воспроизводимым

##2
from reservoirpy.nodes import Reservoir
#Сначала мы создадим резервуар для нашего ESN со 100 нейронами.
#lr: скорость утечки, sr: спектральный радиус рекуррентных соединений в резервуаре

reservoir = Reservoir(100, lr=0.5, sr=0.9)

##3
#массив синуса по времени, 100 значений
import numpy as np
import matplotlib.pyplot as plt

X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)

plt.figure(figsize=(10, 3))
plt.title("A sine wave.")
plt.ylabel("$sin(t)$")
plt.xlabel("$t$")
plt.plot(X)
plt.show()

##4
#могут работать с отдельными временными шагами данных или полными временными рядами
#запустить узел на одном временном шаге данных:
s = reservoir(X[0].reshape(1, -1))

#Новая форма вектора состояния
print("New state vector shape: ", s.shape)

##5
s = reservoir.state()
#Доступ к этому состоянию

##6
#мы можем выполнить последовательные вызовы резервуара, чтобы собрать его активации всего временного ряда:
states = np.empty((len(X), reservoir.output_dim))
for i in range(len(X)):
    states[i] = reservoir(X[i].reshape(1, -1))
    
##7
#мы отобразили активацию 20 нейронов в резервуаре для каждой точки временного ряда
plt.figure(figsize=(10, 3))
plt.title("Activation of 20 reservoir neurons.")
plt.ylabel("$reservoir(sin(t))$")
plt.xlabel("$t$")
plt.plot(states[:, :20])
plt.show()

##8
#Сбор активаций узла по временному ряду можно выполнить без использования цикла for
states = reservoir.run(X)

##9
from reservoirpy.nodes import Reservoir, Ridge, FORCE, ESN

##10
#состояние узла можно сбросить до нулевого вектора
reservoir = reservoir.reset()

##11
#или то же самое так
states_from_null = reservoir.run(X, reset=True)

##12
#Состояния также могут передаваться узлу в любое время с помощью from_state
a_state_vector = np.random.uniform(-1, 1, size=(1, reservoir.output_dim))

states_from_a_starting_state = reservoir.run(X, from_state=a_state_vector)

##13
#Эти операции также можно выполнить без стирания памяти узла с помощью with_state
previous_states = reservoir.run(X)

with reservoir.with_state(reset=True):
    #??
    states_from_null = reservoir.run(X)

# as if the with_state never happened !
states_from_previous = reservoir.run(X)

##<-----здесь
##14
from reservoirpy.nodes import Ridge

readout = Ridge(ridge=1e-7)
#При установке ridgeпараметра считывания на 1e-7. Это регуляризация, гиперпараметр, который поможет избежать переобучения.

##15
#Определите тренировочную задачу
#Подобные узлы Ridge можно обучать с помощью их fit() метода
#два временных ряда: входной временной ряд и целевой временной ряд.
X_train = X[:50]
Y_train = X[1:51]

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(X_train, label="sin(t)", color="blue")
plt.plot(Y_train, label="sin(t+1)", color="red")
plt.legend()
plt.show()

##16
train_states = reservoir.run(X_train, reset=True)

##17
#тренируем
#warmup параметр - установить количество временных шагов, которые мы хотим отбросить
readout = readout.fit(train_states, Y_train, warmup=10)

##18
test_states = reservoir.run(X[50:])
Y_pred = readout.run(test_states)

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(Y_pred, label="Predicted sin(t)", color="blue")
plt.plot(X[51:], label="Real sin(t+1)", color="red")
plt.legend()
plt.show()

##19
#Создайте модель ESN
#ESN — это очень простой тип модели, содержащий два узла: резервуар и вывод.
#Чтобы объявить связи между узлами и построить модель, используйте >> оператор:
from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge

##20
#Тренируйте ESN
esn_model = esn_model.fit(X_train, Y_train, warmup=10)

##21
print(reservoir.is_initialized, readout.is_initialized, readout.fitted)

##22
#Запустите ESN
Y_pred = esn_model.run(X[50:])

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(Y_pred, label="Predicted sin(t+1)", color="blue")
plt.plot(X[51:], label="Real sin(t+1)", color="red")
plt.legend()
plt.show()



























