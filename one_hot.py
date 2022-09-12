# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:00:55 2020

@author: guoyajing
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split

coden_dict = {'A':0,
              'U':1,
              'C':2,
              'G':3,
              }

def coden(seq):
    vectors = np.zeros((len(seq), 4))
    for i in range(len(seq)):
        vectors[i][coden_dict[seq[i].replace('T', 'U')]] = 1
    return vectors.tolist()

def dealwithdata(protein):
    protein = protein
    dataX = []
    dataY = []
    dataYc = []
    with open('../dataset/' + protein + '/positive') as f:
        for line in f:
            line = line.strip('\n')
            if '>' not in line:
                dataX.append(coden(line.strip()))
                dataY.append([0, 1])
                dataYc.append(1)
    with open('../dataset/' + protein + '/negative') as f:
        for line in f:
            line = line.strip('\n')
            if '>' not in line:
                dataX.append(coden(line.strip()))
                dataY.append([1, 0])
                dataYc.append(0)
    indexes = np.random.choice(len(dataY), len(dataY), replace=False)
    dataX = np.array(dataX)[indexes]
    dataY = np.array(dataY)[indexes]
    dataYc = np.array(dataYc)[indexes]
    train_X, test_X, train_y, test_y = train_test_split(dataX, dataY, test_size=0.2, random_state=0)
    train_X1, test_X1, train_y1, test_y1 =  train_test_split(dataX, dataYc, test_size=0.2, random_state=0)
    return train_X, test_X, train_y, test_y, train_y1
