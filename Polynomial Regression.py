#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: griffin
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 1, 1], [1, 2, 4], [1, 3, 9], [1, 4, 16], [1, 5, 25], [1, 6, 36], [1, 7, 49], [1, 8, 64], [1, 21, 441], [1, 25,625], [1, 31, 961], [1, 43, 1849], [1, 55, 3025], [1, 56, 3136], [1, 75, 5625], [1, 80, 6400]]);
#X = np.stack((X0, X1));
#print(X);
Y = np.array([[40], [30], [61], [80], [130], [92], [144], [160], [193], [400], [333], [255], [232], [155], [114], [78]]);
#Y = np.random.randint(-10, 100, 100);
plt.plot(X[:, 1], Y, 'ro')

#h = np.array([[-42.78334132], [0.2296196]]);
theta = np.array([[0], [0], [0]]);
#print(np.sum((X.dot(theta) - Y) * X, 0))
dif = (X.dot(theta) - Y)


#h = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y);
#print(h)
alpha = 0.0000001
#print(theta - np.reshape(alpha * (1/8) * np.sum((X.dot(theta) - Y) * X, 0), (2, 1)))
for i in range(500000):
   theta = theta - np.reshape(alpha * (1/16) * np.sum((X.dot(theta) - Y) * X, 0), (3, 1));
   dif = (X.dot(theta) - Y)

print("Cost function: ", np.dot(dif.T, dif) * 1/16)
print(theta)
#print(np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y));
plt.plot(np.arange(80), (np.reshape(np.arange(80), (80, 1)) * np.reshape(np.arange(80), (80, 1)) * theta[2]) + (np.reshape(np.arange(80), (80, 1)) * theta[1]) + np.full((80,1), theta[0]));
plt.show()

    
