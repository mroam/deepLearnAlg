#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:25:05 2018

@author: griffin
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#imgMat = []
#yVect = []
m = 5000
mt = 5000
paramSize = 65536 * 3

#testMat = []
#yTestVect = []

def loadTrainData(m):
    for i in range(m):
        if ((i % 2) == 0):
            toChoose = "cat"
            yVect.append(1)
        else:
            toChoose = "dog"
            yVect.append(0)
        img = Image.open("train/{0}.{1}.jpg".format(toChoose, i))
        img = img.resize((256, 256))
        imgVector = img.getdata()
        imgVector = np.array(imgVector)
        imgVector = imgVector.reshape((65536*3, 1))
        imgMat.append(imgVector[:, 0])

def loadTestData(mt):
    for i in range(mt):
        if ((i % 2) == 0):
            toChoose = "cat"
            yTestVect.append(1)
        else:
            toChoose = "dog"
            yTestVect.append(0)
        img = Image.open("train/{0}.{1}.jpg".format(toChoose, 10000-i))
        img = img.resize((256, 256))
        imgVector = img.getdata()
        imgVector = np.array(imgVector)
        imgVector = imgVector.reshape((65536*3, 1))
        testMat.append(imgVector[:, 0])

X_train = np.load("X_train_data.npy")

X_test = np.load("X_test_data.npy")

Y_train = np.load("Y_train_data.npy")

Y_test = np.load("Y_test_data.npy")

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape)

def sigmoid(z):
    sZ = 1/(1+np.exp(-z))
    
    return sZ

def init_params(numOfParams):
    b = np.zeros(1)
    w = np.zeros((numOfParams, 1))
    
    return w, b

def prop(w, b, X, Y):
    m = X.shape[1]
    activationFunc = sigmoid(np.dot(w.T, X) + b)
    diff = activationFunc - Y
    db = (1/m) * np.sum(diff)
    dw = (1/m) * np.dot(X, diff.T)
    cost = -(1/m) * np.sum(Y*np.log(activationFunc) + (1-Y)*np.log(1-activationFunc))
    print(cost)
    
    return dw, db

def learn(w, b, X, Y, alpha, numOfIter):
    for i in range(numOfIter):
        dw, db = prop(w, b, X, Y)
        w = w - alpha * dw
        b = b - alpha * db
    
    return w, b

def predictY(w, b, X):
    print(w.shape)
    guess = sigmoid(np.dot(w.T, X) + b)
    
    return guess

w, b = init_params(paramSize);
w, b = learn(w, b, X_train, Y_train, 0.0005, 1);
guess = predictY(w, b, X_train);
print("Training set accuracy: ", (1 - np.mean(np.abs((guess - Y_train))))*100, "%")
guess = predictY(w, b, X_test)
print("Test set accuracy: ", (1 - np.mean(np.abs((guess - Y_test))))*100, "%")


