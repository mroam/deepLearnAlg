#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:26:37 2018

@author: gtier
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def tanh(x):
    t = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    return t

def relu(x):
    r = np.maximum(0, x)
    return r

def set_up_data(m):
    dataSet = sklearn.datasets.make_moons(n_samples=m, noise=0.1)
    #dataSet = sklearn.datasets.make_circles(n_samples=m, noise=0.2, factor = 0.02)
    #dataSet = sklearn.datasets.make_blobs(n_samples=m, centers=10)
    #tempX = dataSet[0]
    #tempY = dataSet[1] % 2
    #dataSet = (tempX, tempY)
    return dataSet


def plot_data(dataSet):
    plt.scatter(dataSet[0][:, 0], dataSet[0][:, 1], c=dataSet[1], cmap=plt.cm.RdBu)
  
#n_h can be changed to alter number of hidden layers (change when calling NN_model)
def init_params(X, n_h):
    W1 = np.random.randn(n_h, np.size(X, axis = 0)) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(1, n_h) * 0.01
    b2 = np.zeros((1,1))
    return W1, b1, W2, b2
    
#A2 is output layer
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


def back_prop(A1, A2, Z1, X, W2, Y, m):
    dZ2 = A2 - Y
    dW2 = (1/m)*dZ2.dot(A1.T)
    db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = W2.T.dot(dZ2)*(1-np.square(A1))
    dW1 = (1/m)*dZ1.dot(X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims = True)
    return dW1, db1, dW2, db2


def predict(X, W1, b1, W2, b2):
    A1, Z1, A2, Z2 = forward_prop(X, W1, b1, W2, b2)
    predictions = np.rint(A2)
    return predictions


def plot_boundary(X, W1, b1, W2, b2, dataSet):
    step = 0.025
    x_min = np.amin(X[0, :]) - 1
    x_max = np.amax(X[0, :]) + 1
    y_min = np.amin(X[1, :]) - 1
    y_max = np.amax(X[1, :]) + 1
    xMesh, yMesh = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    
    #print(np.c_[xMesh.ravel(), yMesh.ravel()].T.shape)
    predictions = predict(np.c_[xMesh.ravel(), yMesh.ravel()].T, W1, b1, W2, b2)
    predictions = predictions.reshape(xMesh.shape)
    #print(predictions.shape)
    plt.contourf(xMesh, yMesh, predictions, cmap=plt.cm.RdBu, alpha=0.7)
    plot_data(dataSet)
    plt.show()
    

def NN_Model(iterations, learning_rate, n_h, m):
    dataSet = set_up_data(m)
    X = dataSet[0].T
    Y = dataSet[1]
    W1, b1, W2, b2 = init_params(X, n_h)
    for i in range(iterations):
        A1, Z1, A2, Z2 = forward_prop(X, W1, b1, W2, b2)
        c = cost(A2, Y, m)
        if((i % 1000) == 0):
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
    plot_boundary(X, W1, b1, W2, b2, dataSet)
    print("Accuracy: " + str(accuracy) + "%")
    
#NN_Model(iterations, learning_rate, number of hidden layers, number of items in dataset)    
NN_Model(10000, 1, 5, 100)

