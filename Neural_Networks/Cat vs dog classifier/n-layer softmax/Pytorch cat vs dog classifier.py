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
        imgVectorCat = np.array(imgVectorCat)
        imgVectorCat = imgVectorCat.reshape((dimension*dimension*3))
        imgMat.append(imgVectorCat)

        toChoose = "dog"
        yVect.append(0)
        imgDog = Image.open(dir_path + "/train/{0}.{1}.jpg".format(toChoose, i))
        imgDog = imgDog.resize((dimension, dimension))
        imgVectorDog = imgDog.getdata()
        imgVectorDog = np.array(imgVectorDog)
        imgVectorDog = imgVectorDog.reshape((dimension*dimension*3))
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
    for i in range(devM):
        if ((i % 2) == 0):
            toChoose = "cat"
            yVect.append(1)
        else:
            toChoose = "dog"
            yVect.append(0)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        img = Image.open(dir_path + "/train/{0}.{1}.jpg".format(toChoose, (12499-i)))
        img = img.resize((dimension, dimension))
        imgVector = img.getdata()
        imgVector = np.array(imgVector)
        imgVector = imgVector.reshape((dimension*dimension*3, 1))
        imgMat.append(imgVector[:, 0])
    dataSet = (np.array(imgMat)/255, np.array(yVect).reshape(devM))
    return dataSet

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
            tempX = np.array(X[lastIndex:i, :])
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
    
main(50, 1000, 2, 24000, 2000, 96)
