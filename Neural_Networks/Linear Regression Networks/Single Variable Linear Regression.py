#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: griffin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 21], [1, 25], [1, 31], [1, 43], [1, 55], [1, 56], [1, 75], [1, 80]]);
#X = np.stack((X0, X1));
#print(X);
Y = np.array([[40], [30], [61], [80], [130], [92], [144], [160], [193], [400], [333], [255], [232], [155], [114], [78]]);
#Y = np.random.randint(-10, 100, 100);
plt.plot(X[:, 1], Y, 'ro')

#h = np.array([[-42.78334132], [0.2296196]]);
theta = np.array([[0], [0]]);
#print(np.sum((X.dot(theta) - Y) * X, 0))
dif = (X.dot(theta) - Y)


#h = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y);
#print(h)
alpha = 0.0007
#print(theta - np.reshape(alpha * (1/8) * np.sum((X.dot(theta) - Y) * X, 0), (2, 1)))
for i in range(10000):
   theta = theta - np.reshape(alpha * (1/8) * np.sum((X.dot(theta) - Y) * X, 0), (2, 1));
   dif = (X.dot(theta) - Y)
   
   #print(theta)

print("Cost function: ",np.dot(dif.T, dif) * 1/16)
print("H: ", theta)
#print(np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y));
plt.plot(np.arange(80), (np.reshape(np.arange(80), (80, 1)) * theta[1]) + np.full((80,1), theta[0]));

#T0list = []
#T1list = []
#Jlist = []
#countW = -5;
#for w in range(10):
#  countZ = -5;
#   for z in range(10):
#       theta[0] = countW;
#       theta[1] = countZ;
#       T0list.append(countW)
#       T1list.append(countZ)
#       dif = (X.dot(theta) - Y)
#       J = np.dot(dif.T, dif) * 1/16;
#       Jlist.append(J[0])
#       countZ += 1;
#   countW += 1;


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#resX = np.asarray(T0list);
#resY = np.asarray(T1list);
#resJ = np.asarray(Jlist);
#resJ = resJ.T * (np.ones(100));
#resX, resY = np.meshgrid(resX, resY)
#print(resX, resY, resJ[0])
#Jplot = ax.plot_surface(resX, resY, resJ, rstride=1, cstride=1, cmap=cm.coolwarm,
#               linewidth=0, antialiased=False)
#fig.colorbar(Jplot, shrink=0.5, aspect=10)

plt.show()

    
