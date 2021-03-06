#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gtier
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

def sigmoid(Z):
    s = 1/(1+np.exp(-Z))
    return s

def set_up_data(m):
    dataSet = sklearn.datasets.make_circles(n_samples=m, noise=0.2, factor=0.2)
    return dataSet

def plot_data(dataSet):
    plt.scatter(dataSet[0][:, 0], dataSet[0][:, 1], c=dataSet[1], cmap=plt.cm.Spectral)
    plt.show()
    
def init_params(X):
    W = np.random.randn(1, np.size(X, axis = 0))
    b = np.zeros((1,1))
    return W, b
    
def cost(A, Y, m):
    c = (-1/m) * np.sum(np.log(A) * Y + (1-Y)*np.log(1-A))
    return c

def forward_prop(X, W, b):
    Z = W.dot(X) + b
    A = sigmoid(Z)
    return A, Z

def back_prop(A, X, Y, m):
    dZ = A - Y
    dW = dZ.dot(X.T)
    db = np.sum(dZ)
    return dW, db

def predict(X, W, b):
    pZ = W.dot(X) + b
    pA = sigmoid(pZ)
    predictions = np.rint(pA)
    return predictions

def plot_boundary(X, W, b, dataSet):
    step = 0.025
    x_min = np.amin(X[0, :]) - 1
    x_max = np.amax(X[0, :]) + 1
    y_min = np.amin(X[1, :]) - 1
    y_max = np.amax(X[1, :]) + 1
    xMesh, yMesh = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    
    print(np.c_[xMesh.ravel(), yMesh.ravel()].T.shape)
    predictions = predict(np.c_[xMesh.ravel(), yMesh.ravel()].T, W, b)
    predictions = predictions.reshape(xMesh.shape)
    print(predictions.shape)
    plt.figure(1)
    plt.contourf(xMesh, yMesh, predictions, cmap=plt.cm.Spectral, alpha=0.75)
    plot_data(dataSet)
    plt.show()
    plt.figure(2)
    plot_data(dataSet)
    plt.show()

def NN_Model(iterations, learning_rate, m):
    dataSet = set_up_data(m)
    X = dataSet[0].T
    Y = dataSet[1]
    W, b = init_params(X)
    for i in range(iterations):
        A, Z = forward_prop(X, W, b)
        c = cost(A, Y, m)
        if((i % 1000) == 0):
            print("Cost after iteration " + str(i) + ": " + str(c))
        dW, db = back_prop(A, X, Y, m)
        W = W - learning_rate * dW
        b = b - learning_rate * db
    print(W, b)
    predictions = predict(X, W, b)
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
    plot_boundary(X, W, b, dataSet)
    print("Accuracy: " + str(accuracy) + "%")
    
    
NN_Model(2000, 0.01, 100)

