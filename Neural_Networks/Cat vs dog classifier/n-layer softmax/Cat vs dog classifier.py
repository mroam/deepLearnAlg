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

def softmax(x):
    temp = np.exp(x)
    softmax = temp/(np.sum(temp, axis = 0))
    return softmax

#Called by set_up_data()
def loadTrainData(m):
    yVect = []
    imgMat = []
    for i in range(m):
        if ((i % 2) == 0):
            toChoose = "cat"
            yVect.append([1, 0])
        else:
            toChoose = "dog"
            yVect.append([0, 1])
        dir_path = os.path.dirname(os.path.realpath(__file__))
        img = Image.open(dir_path + "/train/{0}.{1}.jpg".format(toChoose, i))
        img = img.resize((24, 24))
        imgVector = img.getdata()
        imgVector = np.array(imgVector)
        imgVector = imgVector.reshape((24*24*3))
        imgMat.append(imgVector)
    npData = np.array(imgMat)
    dataSet = (((npData - np.mean(npData))/np.std(npData)), np.array(yVect).T)
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
def init_params(X, n):
    weights = {}
    l = len(n)
    weights["W1"] = np.random.randn(n[0], np.size(X, axis = 0)) * np.sqrt(2/np.size(X, axis = 0))
    weights["b1"] = np.zeros((n[0],1))
    for i in range(1, l):
        weights["W" + str(i+1)] = np.random.randn(n[i], n[i-1]) * np.sqrt(2/n[i-1])
        weights["b" + str(i+1)] = np.zeros((n[i], 1))
    return weights

#A2 is output layer
#Y is a vector containing the correct answers
#m is the number of columns in the training set matrix
#lambd is the regularization parameter
#Regularization penalizes large weights
#Add regularization TODO
def cost(Yhat, Y, m):
    c = (-1/m)*(np.sum(np.sum(np.multiply(Y, np.log(Yhat)), axis = 0)))
    return c


#X is the dataset, W is weight, b is bias
def forward_prop(X, weights):
    cache = {}
    l = len(weights) // 2
    cache["Z1"] = weights["W1"].dot(X) + weights["b1"]
    cache["A1"] = relu(cache["Z1"])
    for i in range(1, l):
        cache["Z" + str(i+1)] = weights["W" + str(i+1)].dot(cache["A" + str(i)]) + weights["b" + str(i + 1)]
        if (i < (l-1)):
            cache["A" + str(i+1)] = relu(cache["Z" + str(i+1)])
        else:
            cache["A" + str(i+1)] = softmax(cache["Z" + str(i+1)])
    return weights, cache

#This is called by NN_Model
def back_prop(learning_rate, X, Y, weights, cache):
    l = len(weights) // 2
    m = np.size(X, axis = 1)
    derivatives = {}
    derivatives["dZ" + str(l)] = cache["A" + str(l)] - Y
    derivatives["dW" + str(l)] = (1/m)*(derivatives["dZ" + str(l)].dot(cache["A" + str(l-1)].T))
    derivatives["db" + str(l)] = (1/m)*np.sum(derivatives["dZ" + str(l)], axis = 1, keepdims = True)
    derivatives["dA" + str(l-1)] = weights["W" + str(l)].T.dot(derivatives["dZ" + str(l)])
    for i in range(l-1, 0, -1):
        derivatives["dZ" + str(i)] = derivatives["dA" + str(i)] * np.int64(cache["A" + str(i)])
        if (i>1):
            derivatives["dW" + str(i)] = (1/m)*(derivatives["dZ" + str(i)].dot(cache["A" + str(i-1)].T))
        else:
            derivatives["dW" + str(i)] = (1/m)*(derivatives["dZ" + str(i)].dot(X.T))
        derivatives["db" + str(i)] = (1/m)*np.sum(derivatives["dZ" + str(i)], axis = 1, keepdims = True)
        if (i > 1):
            derivatives["dA" + str(i-1)] = weights["W" + str(i)].T.dot(derivatives["dZ" + str(i)])
    for i in range(l):
        weights["W" + str(i+1)] = weights["W" + str(i+1)] - learning_rate * derivatives["dW" + str(i+1)]
        weights["b" + str(i+1)] = weights["b" + str(i+1)] - learning_rate * derivatives["db" + str(i+1)]
    return weights

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
def test_accuracy(weights, X, Y):
    weights, cache = forward_prop(X, weights)
    predictions = cache["A" + str(len(weights)//2)]
    accuracy = Y * predictions
    accuracy = np.sum(np.sum(accuracy, axis = 0)) / np.size(X, axis = 1)
    return accuracy



#Main model function
#Yet to make this NN_Model multi-hidden layer TODO
#NN_Model(iterations, learning_rate, number of hidden units in hidden layer, number of items in training set, number of items in dev set)
def NN_Model(iterations, learning_rate, n, m, devM):
    dataSet = set_up_data(m)
    X = dataSet[0].T
    print("X shape: ", X.shape)
    Y = dataSet[1]
    print("Y shape: ", Y.shape)
    l = len(n)
    weights = init_params(X, n)
    for i in range(iterations):
        weights, cache = forward_prop(X, weights)
        if ((i % 1000) == 0):
            print("Cost after iteration " + str(i) + ":", cost(cache["A" + str(l)], Y, m))
        weights = back_prop(learning_rate, X, Y, weights, cache)
    trainAccuracy = test_accuracy(weights, X, Y)
    print("Training set accuracy: ", trainAccuracy * 100 + "%")

#NN_Model(iterations, learning_rate, number of hidden units in hidden layer, number of items in training set, number of items in dev set)
NN_Model(20000, 0.00005, [10,10, 2], 1000, 2000)
