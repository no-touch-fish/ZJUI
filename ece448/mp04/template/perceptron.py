# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020



import numpy as np

def trainPerceptron(train_set, train_labels,  max_iter):
    #first get the dimension and init
    k = 0.01 # learning rate
    data_num, feature_num = train_set.shape
    w = np.zeros(feature_num)
    b = np.zeros(1)
    # train
    for i in range(max_iter):
        for index in range(data_num):
            # classifier output
            predict = w @ train_set[index, :] + b
            if (predict <= 0):
                predict = 0
            else:
                predict = 1
            # update weight vector
            if (predict != train_labels[index]):
                if (predict == 0):
                    w += k * train_set[index,:]
                    b += k
                else:
                    w -= k * train_set[index,:]
                    b -= k
    return w, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4
    data_num = len(dev_set)
    output = []
    w,b = trainPerceptron(train_set,train_labels,max_iter)
    for index in range(data_num):
        if (w @ dev_set[index,:] + b <= 0):
            output.append(0)
        else:
            output.append(1)
    return output



