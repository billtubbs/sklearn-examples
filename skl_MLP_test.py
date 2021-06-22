# Demonstration of how to change the output activation
# function in a multi-layer perceptron regression model
# Thanks to Ggjj11 for this stackexchange answer:
# See https://datascience.stackexchange.com/a/96937

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network._base import ACTIVATIONS, DERIVATIVES


rng = np.random.RandomState()

def _initialize(self, *args, **kwargs):
    self._old_initialize(*args, **kwargs)
    self.out_activation_ = "tanh"


# Standard MLPRegressor models
model1 = MLPRegressor(hidden_layer_sizes=(1), solver='lbfgs',
                      random_state=rng)
model2 = MLPRegressor(hidden_layer_sizes=(1), solver='lbfgs',
                      random_state=rng)

# Modification to specify the output layer
# activation function.
model2._old_initialize = model2._initialize
model2._initialize = _initialize.__get__(model2)

# Test models on simple system
f = lambda x: np.tanh(x*4)

# Training data (noisy)
X = 4*rng.random(size=100) - 2
y = f(X) + rng.randn(100)*0.1

# Test data
X_test = np.linspace(-2, 2, 11)
y_test = f(X_test)

model1.fit(X.reshape(-1, 1), y, )
print(f"Model 1 score: {model1.score(X_test.reshape(-1, 1), y_test)}")

model2.fit(X.reshape(-1, 1), y)
#print(f"Model 2 score: {model2.score(X_test.reshape(-1, 1), y_test)}")

X_pred = np.linspace(-2, 2, 101)
y_pred1 = model1.predict(X_pred.reshape(-1, 1))
y_pred2 = model2.predict(X_pred.reshape(-1, 1))

plt.plot(X, y, '.')
plt.plot(X_pred, y_pred1, '-', label='standard model')
plt.plot(X_pred, y_pred2, '-', label='modified model')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()
