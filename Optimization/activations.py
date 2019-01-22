import numpy as np
import matplotlib.pylab as plt
plt.ion()

def threshold(x):
    return (x / np.abs(x)/2) + 1

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.max(np.vstack([x, np.zeros(len(x))]), axis=0)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x):
    sign = x/np.abs(x)

    return (1-sign)*x/2 * 0.1 + (1+sign)*x/2

def exponential_leaky_relu(x):
    sign = x/np.abs(x)

    return (1-sign)/2*(np.exp(x)-1) + (1+sign)*x/2    

def plot(x, func, titlename, filename=None):
    plt.figure(figsize=(10,8))

    y = func(x)

    plt.plot(x,y)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.title(f'Activation Function: {titlename}')

    if filename is not None:
        plt.savefig(filename, transparent=True, bbox_inches='tight')

if __name__=='__main__':
    x = np.arange(-10, 10, 0.1)
    for name in ['threshold', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'exponential_leaky_relu']:
        plot(x, locals()[name], name, f'plots/{name}.png')
