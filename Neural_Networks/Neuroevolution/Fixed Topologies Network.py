#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:18:11 2018

@author: gtier
"""
import time
import gym
import numpy as np
import math
import random
import copy

def relu(x):
    r = np.maximum(0, x)
    return r

def tanh(x):
    t = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    return t

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def init_params(env, layers, population):
    paramsList = []
    for p in range(population):
        params = {}
        for i in range(len(layers)):
            if (i==0):
                weight1 = np.random.randn(layers[0], env.observation_space.shape[0])
                bias1 = np.zeros((layers[0], 1))
                params = { "weight1":weight1, "bias1":bias1}
            else:
                tempWeight = np.random.randn(layers[i], layers[i-1])
                tempBias = np.zeros((layers[i], 1))
                params["weight" + str(i + 1)] = tempWeight 
                params["bias" + str(i + 1)] = tempBias
        paramsList.append(params)
        #for key in params:
         #   print(key, params[key].shape)
    return paramsList

def forward_prop(state, params):
    activations = {}
    length = (len(params) // 2)
    output = []
    for i in range(length):
        if (i==0):
            temp = params["weight1"].dot(state) + params["bias1"]
            activations["activations1"] = tanh(temp)
        elif (i < length - 1):
            temp = params["weight" + str(i + 1)].dot(activations["activations" + str(i)]) + params["bias" + str(i + 1)]
            activations["activations" + str(i + 1)] = tanh(temp)
        else:
            temp = params["weight" + str(i + 1)].dot(activations["activations" + str(i)]) + params["bias" + str(i + 1)]
            output = sigmoid(temp)
    
    output = np.argmax(output)
    return output, activations

def mutate(paramList, population):
    tempList=[]
    for val in paramList:
        tempList.append(copy.deepcopy(val))
    for val in paramList:
        for t in range(999):
            temp = copy.deepcopy(val)
            randomKey = random.choice(list(temp.keys()))
            temp[randomKey] = temp[randomKey] + math.pow(-1, random.randint(1,2)) * np.random.randn(*temp[randomKey].shape) * 0.1
            #for i in range(random.randint(5, 25)):
                #randomKey = random.choice(list(temp.keys()))
                #randomMutation = random.randint(0, 20)
                #if (randomMutation == 20):
                   # tempRandn = np.random.randn(1,1)
                    #temp[randomKey] = temp[randomKey] * np.amax(tempRandn) works well with MountainCar-v0
                    #temp[randomKey] = temp[randomKey] * np.amax(tempRandn) * 0.01
                #else:
                    #temp[randomKey] = (temp[randomKey] + math.pow(-1, random.randint(1, 2)) * random.random() * 0.1)
                    #temp[randomKey] = (temp[randomKey] + math.pow(-1, random.randint(1, 2)) * random.random() * 0.01)
            tempList.append(temp)
    return tempList

def train(env, generations, iterations, population, layers):
    currLeader = {}
    currMaxScore = -201
    bestOfLastGen = []
    bestScoresOfLastGen = []
    for genNum in range(generations):
        if (genNum == 0):
            populationParams = init_params(env, layers, population)
        else:
            populationParams = mutate(bestOfLastGen, population)
        populationScores = []
        for params in populationParams:
            done = False
            additiveScore = 0
            state = env.reset()
            state = state.reshape(env.observation_space.shape[0], 1)
            for t in range(iterations):
                output, activations = forward_prop(state, params)
                state, reward, done, info = env.step(output)
                additiveScore += reward
                state = state.reshape(env.observation_space.shape[0], 1)
                if (done == True):
                    populationScores.append(additiveScore)
                    #populationScores.append(reward)
                    break
        topTenPercentScores = np.argpartition(populationScores, (-population//1000))[(-population//1000):]
        for val in topTenPercentScores:
            bestOfLastGen = []
            bestScoresOfLastGen = []
            bestOfLastGen.append(populationParams[val])
           # print(populationScores)
           # print("Val", val)
            bestScoresOfLastGen.append(populationScores[val])
        print("Mean score in generation " + str(genNum) + ": " + str(np.mean(populationScores)))
        print("Max score in generation " + str(genNum) + ": " + str(np.amax(populationScores)))
    print("Final average score :", np.mean(bestScoresOfLastGen))
    print("Final best score", np.amax(bestScoresOfLastGen))
    currLeader = bestOfLastGen[np.argmax(bestScoresOfLastGen)]
    return currLeader

def load_environment():
    env = gym.make("MountainCar-v0") #CartPole-v1 #BipedalWalker-v2 #LunarLander-v2 #MountainCar-v0
    print("Observation_space: ", env.observation_space)
    print("Action_space: ", env.action_space)
    return env

def test_network(env, params):
    additiveScore = 0
    for p in range(10):
        state = env.reset()
        state = state.reshape(env.observation_space.shape[0], 1)
        done = False
        for t in range(1600):
            output, activations = forward_prop(state, params)
            state, reward, done, info = env.step(output)
            additiveScore += reward
            state = state.reshape(env.observation_space.shape[0], 1)
            env.render()
            if (done == True):
                break
    return additiveScore/10

def main():
    #a = np.load("Neuroevolution_best_weights.npy")
   #print(a)
    env = load_environment()
    #currLeader = train(env, 1, 1600, 1000, [4, 5, 4])
    currLeader = train(env, 5, 1600, 1000, [2, 5, 3]) #works well for MountainCar-v0
    #currLeader = train(env, 1, 1600, 1000, [2, 4, 2]) #works well for CartPole-v1
    finalScore = test_network(env, currLeader)
    print("Final network score: ", finalScore)

main()
    