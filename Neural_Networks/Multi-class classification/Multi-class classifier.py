import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

class NN_Model(nn.Module):

    def __init__(self, Xsize, numOfClasses):
        super(NN_Model, self).__init__()
        self.linearForward1 = nn.Linear(Xsize, 100)
        self.linearForward2 = nn.Linear(100, 50)
        self.linearForward3 = nn.Linear(50, 10)
        self.linearForward4 = nn.Linear(10, numOfClasses)

    def forward(self, X):
        x = self.linearForward1(X)
        x = F.relu(x)
        x = self.linearForward2(x)
        x = F.relu(x)
        x = self.linearForward3(x)
        x = F.relu(x)
        x = self.linearForward4(x)
        return F.softmax(x, dim=1)

def set_up_data(numOfClasses, m):
    dataSet = sklearn.datasets.make_moons(n_samples=m, noise=0.1)
    dataSet = sklearn.datasets.make_circles(n_samples=m, noise=0.2, factor = 0.02)
    dataSet = sklearn.datasets.make_gaussian_quantiles(n_samples=m, n_classes=numOfClasses)
    dataSet = sklearn.datasets.make_blobs(n_samples = m, centers = numOfClasses, n_features=2)
    return dataSet

def test_accuracy(nn_model, X, Y):
    nn_model.zero_grad()
    X = torch.FloatTensor(X)
    output = nn_model(X)
    output = output.detach().numpy()
    output = np.argmax(output, axis=1)
    right = 0
    total = len(output)
    for i in range(total):
        if(output[i] == Y[i]):
            right += 1
            #print(output[i], Y[i])
    avg = right/total
    return avg

def plot_boundary(X, nn_model, dataSet, numOfClasses):
    step = 0.025
    x_min = np.amin(X[:, 0]) - 1
    x_max = np.amax(X[:, 0]) + 1
    y_min = np.amin(X[:, 1]) - 1
    y_max = np.amax(X[:, 1]) + 1
    xMesh, yMesh = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    
    #print(np.c_[xMesh.ravel(), yMesh.ravel()].T.shape)
    predictions = nn_model(torch.FloatTensor(np.c_[xMesh.ravel(), yMesh.ravel()]))
    predictions = np.argmax(predictions.detach().numpy(), axis = 1)
    predictions = predictions.reshape(xMesh.shape)
    print(predictions.shape)
    plt.figure(2)
    plt.contourf(xMesh, yMesh, predictions, c=dataSet[1]/numOfClasses, cmap=plt.cm.Spectral, alpha=0.75)
    plot_data(dataSet, numOfClasses)
    plt.show()

def plot_data(dataSet, numOfClasses):
    plt.scatter(dataSet[0][:, 0], dataSet[0][:, 1], linewidths=1, edgecolors='k', c=dataSet[1]/numOfClasses, cmap=plt.cm.Spectral)

def main(epochs, batch_size, numOfClasses, m):
    dataSet = set_up_data(numOfClasses, m)
    plt.figure(1)
    plot_data(dataSet, numOfClasses)
    X = torch.FloatTensor(dataSet[0])
    m = np.size(X, axis=0)
    Xsize = np.size(X, axis=1)

    nn_model = NN_Model(Xsize, numOfClasses)
    cost_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters())

    print("X shape: ", X.shape)
    Y = torch.LongTensor(dataSet[1])
    print("Y shape: ", Y.shape)
    print("Y: ",Y)
    for epoch in range(epochs):
        nn_model.zero_grad()
        output = nn_model(X)
        cost = cost_function(output, Y)
        cost.backward()
        optimizer.step()
        if((epoch % 100) == 0):
            print("Cost after epoch " + str(epoch) + ":", cost)
    trainAccuracy = test_accuracy(nn_model, X, Y)
    print("Training set accuracy: ", trainAccuracy * 100, "%")
    plot_boundary(X.detach().numpy(), nn_model, dataSet, numOfClasses)
    plt.show()
    
main(2000,100, 5, 100)
