import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
from PIL import Image

class NN_Model(nn.Module):

    def __init__(self, Xsize, numOfClasses):
        super(NN_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.linearForward1 = nn.Linear(64*7*7, 1024)
        self.linearForward2 = nn.Linear(1024, 512)
        self.linearForward3 = nn.Linear(512, 10)
        self.linearForward4 = nn.Linear(10, numOfClasses)

    def forward(self, X):
        x = self.conv1(X)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = x.view(-1, 64*7*7)
        x = self.linearForward1(x)
        x = F.relu(x)
        x = self.linearForward2(x)
        x = F.relu(x)
        x = self.linearForward3(x)
        x = F.relu(x)
        x = self.linearForward4(x)
        return F.softmax(x, dim=1)

def set_up_data():
    dataSet = input_data.read_data_sets("MNIST_data", one_hot=False)
    dataSet = dataSet
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

def main(epochs, batch_size, numOfClasses, loadPic):
    dataSet = set_up_data()
    X = np.array(dataSet.train.images).reshape(55000, 1, 28,28)
    print(X.shape)
    m = np.size(X, axis=0)
    Xsize = np.size(X, axis=1)

    nn_model = NN_Model(Xsize, numOfClasses)
    cost_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters())

    print("X shape: ", X.shape)
    Y = np.array(dataSet.train.labels)
    print("Y shape: ", Y.shape)
    print("Y: ",Y)
    lastIndex = 0
    batchNum = 0
    Xbatch = {}
    Ybatch = {}
    for i in range(1, m):
        if (((i % batch_size) == 0)):
            tempX = np.array(X[lastIndex:i, :, :, :])
            tempY = np.array(Y[lastIndex:i])
            Xbatch["batch" + str(batchNum)] = torch.FloatTensor(tempX)
            Ybatch["batch" + str(batchNum)] = torch.LongTensor(tempY)
            lastIndex = i
            batchNum += 1
    print("Xbatch0 size: ", Xbatch["batch0"].shape)
    print("Ybatch0 size: ", Ybatch["batch0"].shape)
    for epoch in range(epochs):
        cost = 0
        for key in Xbatch:
            nn_model.zero_grad()
            output = nn_model(Xbatch[key])
            cost = cost_function(output, Ybatch[key])
            cost.backward()
            optimizer.step()
        print("Cost after epoch " + str(epoch) + ":", cost)
    trainAccuracy = test_accuracy(nn_model, X, Y)
    print("Training set accuracy: ", trainAccuracy * 100, "%")
    Xtest = np.array(dataSet.test.images)
    Ytest = np.array(dataSet.test.labels)
    testAccuracy = test_accuracy(nn_model, Xtest, Ytest)
    print("Test set accuracy: ", testAccuracy * 100, "%")
    
main(15,100, 10, True)
