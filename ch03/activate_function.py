import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype = np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# step function
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)

# sigmoid function
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y, linestyle = "--")
plt.ylim(-0.1, 1.1)

# ReLU function
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y, linestyle = "-.")
plt.ylim(-0.1, 1.1)
plt.show()
