import reservoirpy as rpy

rpy.verbosity(0)  # no need to be too verbose here
rpy.set_seed(42)  # make everyhting reproducible !

import numpy as np
import matplotlib.pyplot as plt

X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)

X_train = X[:50]
Y_train = X[1:51]

plt.figure(figsize=(10, 3))
plt.title("A sine wave.")
plt.ylabel("$sin(t)$")
plt.xlabel("$t$")
plt.plot(X)
plt.show()
##
from reservoirpy.nodes import Reservoir, Ridge, Input

data = Input()
reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)

esn_model = data >> reservoir >> readout & data >> readout


connection_1 = data >> reservoir >> readout
connection_2 = data >> readout
esn_model = connection_1 & connection_2

esn_model = [data, data >> reservoir] >> readout
##
from reservoirpy.nodes import Reservoir, Ridge, Input, Concat

data = Input()
reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)
concatenate = Concat()

esn_model = [data, data >> reservoir] >> concatenate >> readout
##
from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)

reservoir <<= readout

esn_model = reservoir >> readout

esn_model = esn_model.fit(X_train, Y_train)
esn_model(X[0].reshape(1, -1))

print("Feedback received (reservoir):", reservoir.feedback())
print("State sent: (readout):", readout.state())
##
random_feedback = np.random.normal(0, 1, size=(1, readout.output_dim))

with reservoir.with_feedback(random_feedback):
    reservoir(X[0].reshape(1, -1))

##Генерация и долгосрочное прогнозирование

from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge

esn_model = esn_model.fit(X_train, Y_train, warmup=10)
warmup_y = esn_model.run(X_train[:-10], reset=True)

Y_pred = np.empty((100, 1))
x = warmup_y[-1].reshape(1, -1)

for i in range(100):
    x = esn_model(x)
    Y_pred[i] = x

plt.figure(figsize=(10, 3))
plt.title("100 timesteps of a sine wave.")
plt.xlabel("$t$")
plt.plot(Y_pred, label="Generated sin(t)")
plt.legend()
plt.show()































