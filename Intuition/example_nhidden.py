import torch
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

plt.ion()

L = 10

def generate_data(L=10, stepsize=0.1):
    x = np.arange(-L, L, stepsize)
    y = np.sin(3*x) * np.exp(-x / 8.)
    #y = np.sin(x)

    return x, y

def save_plot():
    plt.clf()
    x,y = generate_data()
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.title(r'$\sin(3x) e^{-x/8}$')
    plt.title(r'$\sin(x)$')

    plt.savefig('plot_data_sin.png', transparent=True, bbox_inches='tight')

class NN(nn.Module):
    def __init__(self, N_input=1, N_output=1, N_hidden_layers=1, N_hidden_nodes=10):
        super(NN, self).__init__()

        self.linear1 = nn.Linear(N_input, N_hidden_nodes)
        self.hidden = nn.ModuleList([])
        for i in range(N_hidden_layers-1):
            self.hidden.append(nn.Linear(N_hidden_nodes, N_hidden_nodes))
        self.linear2 = nn.Linear(N_hidden_nodes, N_output)  

        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.activation(self.linear1(x))
        for layer in self.hidden:
            out = self.activation(layer(out))
        out = self.linear2(out)

        return out

    def binary_forward(self, x):
        out = self.activation(self.linear1(x))
        out[out < 0.5] = 0
        out[out >= 0.5] = 1

        for layer in self.hidden:
            out = layer(out)
            out[out < 0.5] = 0
            out[out >= 0.5] = 1
        out = self.linear2(out)

    def binary_forward2(self, x):
        out = self.activation(self.linear1(x))
        for layer in self.hidden:
            out = self.activation(layer(out))
        
        out[out < 0.5] = 0
        out[out >= 0.5] = 1        
        out = self.linear2(out)

        return out

def train_model(features, target, model, lr, N_epochs, shuffle=False):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(N_epochs):
        if shuffle: #should have no effect on gradients
            indices = torch.randperm(len(features))

            features_shuffled = features[indices]
            target_shuffled = target[indices]
        else:
            features_shuffled = features
            target_shuffled = target

        out = model(features_shuffled)
        loss = criterion(out, target_shuffled)

        if epoch % 1000 == 0:
            print(f'epoch = {epoch} loss = {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

def run(N_hidden_nodes=2, N_hidden_layers=1, N_epochs=20000):
    x,y = generate_data()

    x = torch.Tensor(x).reshape(len(x),1)
    y = torch.Tensor(y).reshape(len(y),1)

    model = NN(N_hidden_layers=N_hidden_layers, N_hidden_nodes=N_hidden_nodes)
    model = train_model(x, y, model, 1e-3, N_epochs)
    
    a = x.reshape(len(x)).detach().numpy()
    b = model(x).reshape(len(x)).detach().numpy()
    c = model.binary_forward(x).reshape(len(x)).detach().numpy()
    d = model.binary_forward2(x).reshape(len(x)).detach().numpy()
    
    plt.clf()
    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.plot(a,b)
    plt.plot(a,c)
    plt.plot(a,d)

    return x,y,model

def introspect(model, x, y):
    params = list(model.parameters())

    thresh = -params[1] / params[0].reshape(len(params[0]))

    thresh_sorted = thresh.sort()

    plt.plot(thresh_sorted[0].detach().numpy(), 'p')
    plt.plot([-L]*len(thresh_sorted[0])) 
    plt.plot([L]*len(thresh_sorted[0])) 
