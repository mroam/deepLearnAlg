#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: gtier

To use this NN call the NN_Model function.
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from PIL import Image
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data

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

#This is called by NN_Model
def set_up_data():
    dataSet = input_data.read_data_sets("MNIST_data", one_hot=True)
    dataSet = dataSet
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

#Called by NN_Model
#Takes a trained NN and tests its accuracy on the dev set.
def test_accuracy(weights, X, Y):
    weights, cache = forward_prop(X, weights)
    predictions = cache["A" + str(len(weights)//2)]
    predictions = np.int64(predictions == np.amax(predictions, axis =0))
    accuracy = Y * predictions
    accuracy = np.sum(np.sum(accuracy, axis = 0)) / np.size(X, axis = 1)
    return accuracy

def save_weights(weights):
    for key in weights:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        fileName = dir_path + "/" + key
        np.save(fileName, weights[key])

def load_saved_weights():
    weights = {}
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/weights"
    for file in os.listdir(dir_path):
        fileName = file[:-4]
        weights[fileName] = np.load(dir_path + "/" + file)
    return weights

#Main model function
#Yet to make this NN_Model multi-hidden layer TODO
#NN_Model(iterations, learning_rate, number of hidden units in hidden layer, number of items in training set, number of items in dev set)
def NN_Model(iterations, learning_rate, n, batch_size):
    dataSet = set_up_data()
    X = np.array(dataSet.train.images).T
    m = np.size(X, axis=1)
    print("X shape: ", X.shape)
    Y = np.array(dataSet.train.labels).T
    print("Y shape: ", Y.shape)
    l = len(n)
    weights = init_params(X, n)
    lastIndex = 0
    batchNum = 0
    Xbatch = {}
    Ybatch = {}
    for i in range(1, m):
        if (((i % batch_size) == 0)):
            Xbatch["batch" + str(batchNum)] = X[:, lastIndex:i]
            Ybatch["batch" + str(batchNum)] = Y[:, lastIndex:i]
            lastIndex = i
            batchNum += 1
    print("Xbatch0 size: ", Xbatch["batch0"].shape)
    print("Ybatch0 size: ", Ybatch["batch0"].shape)
    for i in range(iterations):
        batchCount = 0
        costList = []
        for key in Xbatch:
            weights, cache = forward_prop(Xbatch[key], weights)
            currCost = cost(cache["A" + str(l)], Ybatch[key], batch_size)
            costList.append(currCost)
            if (((i % 10) == 0) & ((batchCount % 50) == 0)):
                print("Cost after iteration " + str(i) + ":", currCost)
            weights = back_prop(learning_rate, Xbatch[key], Ybatch[key], weights, cache)
            batchCount += 1
        avgCost = np.mean(np.array(costList))
        if ((i % 10) == 0):
            print("Average cost after iterations " + str(i) + ":", avgCost)
    trainAccuracy = test_accuracy(weights, X, Y)
    print("Training set accuracy: ", trainAccuracy * 100, "%")
    Xtest = np.array(dataSet.test.images).T
    Ytest = np.array(dataSet.test.labels).T
    testAccuracy = test_accuracy(weights, Xtest, Ytest)
    print("Test set accuracy: ", testAccuracy * 100, "%")
    #save_weights(weights)

def Run_Saved_Model():
    dataSet = set_up_data()
    print(dataSet.test.images)
    X = np.array(dataSet.train.images).T
    m = np.size(X, axis=1)
    print("X shape: ", X.shape)
    Y = np.array(dataSet.train.labels).T
    print("Y shape: ", Y.shape)
    weights = load_saved_weights()
    trainAccuracy = test_accuracy(weights, X, Y)
    print("Training set accuracy: ", trainAccuracy * 100, "%")
    Xtest = np.array(dataSet.test.images).T
    Ytest = np.array(dataSet.test.labels).T
    testAccuracy = test_accuracy(weights, Xtest, Ytest)
    print("Test set accuracy: ", testAccuracy * 100, "%")

#NN_Model(iterations, learning_rate, number of hidden units in hidden layer, batch_size)
NN_Model(1000, 0.05, [10, 10], 1000)
#Run_Saved_Model()