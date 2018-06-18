#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: gtier

To use this NN call the NN_Model function.
Also, to use this NN put the /train and /test1 folder in the same directory as the Cat vs dog classifier.
Note: The dev set should not be the same as the test set.
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from PIL import Image
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Called by set_up_data()
def loadTrainData(m, dimension):
    yVect = []
    imgMat = []
    for i in range(m // 2):

        toChoose = "cat"
        yVect.append(1)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        imgCat = Image.open(dir_path + "/train/{0}.{1}.jpg".format(toChoose, i))
        imgCat = imgCat.resize((dimension, dimension))
        imgVectorCat = imgCat.getdata()
        imgVectorCat = np.array(imgVectorCat).reshape(3, dimension, dimension)
       #print("IMG VECTOR",imgVectorCat.shape)
        imgMat.append(imgVectorCat)

        toChoose = "dog"
        yVect.append(0)
        imgDog = Image.open(dir_path + "/train/{0}.{1}.jpg".format(toChoose, i))
        imgDog = imgDog.resize((dimension, dimension))
        imgVectorDog = imgDog.getdata()
        imgVectorDog = np.array(imgVectorDog).reshape(3, dimension, dimension)
        imgMat.append(imgVectorDog)

    npData = np.array(imgMat)
    dataSet = (((npData - np.mean(npData))/np.std(npData)), np.array(yVect))
    return dataSet

#This is called by NN_Model
def set_up_data(m, dimension):
    dataSet = loadTrainData(m, dimension)
    #dataSet = sklearn.datasets.make_moons(n_samples=m, noise=0.1)
    #dataSet = sklearn.datasets.make_circles(n_samples=m, noise=0.2, factor = 0.02)
    #dataSet = sklearn.datasets.make_blobs(n_samples=m, centers=10)
    #tempX = dataSet[0]
    #tempY = dataSet[1] % 2
    #dataSet = (tempX, tempY)
    return dataSet

#Called by NN_Model
def loadTestData(m, devM, dimension):
    yVect = []
    imgMat = []
    for i in range(m // 2):

        toChoose = "cat"
        yVect.append(1)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        imgCat = Image.open(dir_path + "/train/{0}.{1}.jpg".format(toChoose, (12499-i)))
        imgCat = imgCat.resize((dimension, dimension))
        imgVectorCat = imgCat.getdata()
        imgVectorCat = np.array(imgVectorCat).reshape(3, dimension, dimension)
       #print("IMG VECTOR",imgVectorCat.shape)
        imgMat.append(imgVectorCat)

        toChoose = "dog"
        yVect.append(0)
        imgDog = Image.open(dir_path + "/train/{0}.{1}.jpg".format(toChoose, (12499-i)))
        imgDog = imgDog.resize((dimension, dimension))
        imgVectorDog = imgDog.getdata()
        imgVectorDog = np.array(imgVectorDog).reshape(3, dimension, dimension)
        imgMat.append(imgVectorDog)

    npData = np.array(imgMat)
    dataSet = (((npData - np.mean(npData))/np.std(npData)), np.array(yVect))
    return dataSet

class NN_Model(nn.Module):

    def __init__(self, Xsize, numOfClasses):
        super(NN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 5, stride=1)
        self.conv2 = nn.Conv2d(5, 10, 5, stride=1)
        self.conv3 = nn.Conv2d(10, 16, 3)
        self.conv4 = nn.Conv2d(16, 20, 3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.linearForward1 = nn.Linear(20*8*8, 100)
        self.linearForward2 = nn.Linear(100, 50)
        self.linearForward3 = nn.Linear(50, 10)
        self.linearForward4 = nn.Linear(10, numOfClasses)

    def forward(self, X):
        x = self.conv1(X)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = x.view(-1, 20*8*8)
        x = self.linearForward1(x)
        x = F.relu(x)
        x = self.linearForward2(x)
        x = F.relu(x)
        x = self.linearForward3(x)
        x = F.relu(x)
        x = self.linearForward4(x)
        return F.softmax(x, dim=1)

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

def main(epochs, batch_size, numOfClasses, m, devM, dimension):
    dataSet = set_up_data(m, dimension)
    X = dataSet[0]
    m = np.size(X, axis=0)
    Xsize = np.size(X, axis=1)

    nn_model = NN_Model(Xsize, numOfClasses)
    cost_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters())

    print("X shape: ", X.shape)
    Y = dataSet[1]
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
    testSet = loadTestData(m, devM, dimension)
    Xtest = testSet[0]
    Ytest = testSet[1]
    testAccuracy = test_accuracy(nn_model, Xtest, Ytest)
    print("Test set accuracy: ", testAccuracy * 100, "%")
    
main(20, 100, 2, 20000, 2000, 48)
