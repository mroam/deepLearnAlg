#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: gtier

To use this NN call the NN_Model function.
Also, to use this NN put the /train and /test1 folder in the same directory as the Cat and dog classifier.
Note: The dev set should not be the same as the test set.
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from PIL import Image
import os
import sys

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def tanh(x):
    t = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    return t

#Relu is faster than tanh, but I havn't implemented it yet. TODO
def relu(x):
    r = np.maximum(0, x)
    return r

#Called by set_up_data()
def loadTrainData(m):
    yVect = []
    imgMat = []
    for i in range(m):
        if ((i % 2) == 0):
            toChoose = "cat"
            yVect.append(1)
        else:
            toChoose = "dog"
            yVect.append(0)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        img = Image.open(dir_path + "/train/{0}.{1}.jpg".format(toChoose, i))
        img = img.resize((24, 24))
        imgVector = img.getdata()
        imgVector = np.array(imgVector)
        imgVector = imgVector.reshape((24*24*3, 1))
        imgMat.append(imgVector[:, 0])
    dataSet = (np.array(imgMat)/255, np.array(yVect).reshape(m))
    return dataSet

#This is called by NN_Model
def set_up_data(m):
    dataSet = loadTrainData(m)
    #dataSet = sklearn.datasets.make_moons(n_samples=m, noise=0.1)
    #dataSet = sklearn.datasets.make_circles(n_samples=m, noise=0.2, factor = 0.02)
    #dataSet = sklearn.datasets.make_blobs(n_samples=m, centers=10)
    #tempX = dataSet[0]
    #tempY = dataSet[1] % 2
    #dataSet = (tempX, tempY)
    return dataSet

#n_h can be changed to alter number of hidden layers (change when calling NN_model)
def init_params(X, n_h):
    W1 = np.random.randn(n_h, np.size(X, axis = 0)) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(1, n_h) * 0.01
    b2 = np.zeros((1,1))
    return W1, b1, W2, b2

#A2 is output layer
#Y is a vector containing the correct answers
#m is the number of columns in the training set matrix
#lambd is the regularization parameter
#Regularization penalizes large weights
#Add regularization TODO
def cost(A2, Y, m):
    c = (-1/m) * np.sum(np.log(A2) * Y + (1-Y)*np.log(1-A2))
    return c


#X is the dataset, W is weight, b is bias
def forward_prop(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = tanh(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    return A1, Z1, A2, Z2

#This is called by NN_Model
def back_prop(A1, A2, Z1, X, W2, Y, m):
    dZ2 = A2 - Y
    dW2 = (1/m)*dZ2.dot(A1.T)
    db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = W2.T.dot(dZ2)*(1-np.square(A1))
    dW1 = (1/m)*dZ1.dot(X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims = True)
    return dW1, db1, dW2, db2

#This is called by NN_Model
def predict(X, W1, b1, W2, b2):
    A1, Z1, A2, Z2 = forward_prop(X, W1, b1, W2, b2)
    predictions = np.rint(A2)
    return predictions

#Called by NN_Model
def loadTestData(m, devM):
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
        img = img.resize((24, 24))
        imgVector = img.getdata()
        imgVector = np.array(imgVector)
        imgVector = imgVector.reshape((24*24*3, 1))
        imgMat.append(imgVector[:, 0])
    dataSet = (np.array(imgMat)/255, np.array(yVect).reshape(devM))
    return dataSet

#Called by NN_Model
#Takes a trained NN and tests its accuracy on the dev set.
def test_accuracy(testDataX, testDataY, W1, b1, W2, b2, m, devM):
    predictions = predict(testDataX, W1, b1, W2, b2)
    right = 0
    total = 0
    #print(predictions[0, 1])
    for i in range(devM):
        if (predictions[0, i] == testDataY[i]):
            right += 1
            total += 1
        else:
            total +=1
    accuracy = (right/total) * 100
    return accuracy



#Main model function
#Yet to make this NN_Model multi-hidden layer TODO
#NN_Model(iterations, learning_rate, number of hidden units in hidden layer, number of items in training set, number of items in dev set)
def NN_Model(iterations, learning_rate, n_h, m, devM):
    dataSet = set_up_data(m)
    X = dataSet[0].T
    Y = dataSet[1]
    W1, b1, W2, b2 = init_params(X, n_h)
    print("X", X.shape, Y.shape, W1.shape, W2.shape)
    print("Y", Y)
    for i in range(iterations):
        A1, Z1, A2, Z2 = forward_prop(X, W1, b1, W2, b2)
        c = cost(A2, Y, m)
        if((i % 100) == 0):
            print("Cost after iteration " + str(i) + ": " + str(c))

        dW1, db1, dW2, db2 = back_prop(A1, A2, Z1, X, W2, Y, m)
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

    #print(W1, b1, W2, b2)
    predictions = predict(X, W1, b1, W2, b2)
    right = 0
    total = 0
    #print(predictions[0, 1])
    for i in range(m):
        if (predictions[0, i] == Y[i]):
            right += 1
            total += 1
        else:
            total +=1

    accuracy = (right/total) * 100
    #plot_boundary(X, W1, b1, W2, b2, dataSet)
    print("Train accuracy: " + str(accuracy) + "%")
    testData = loadTestData(m, devM)
    testDataX = testData[0].T
    testDataY = testData[1]
    testAccuracy = test_accuracy(testDataX, testDataY, W1, b1, W2, b2, m, devM)
    print("Test accuracy: " + str(testAccuracy) + "%")

#NN_Model(iterations, learning_rate, number of hidden units in hidden layer, number of items in training set, number of items in dev set)
NN_Model(20000, 0.04, 25, 10000, 2000)
