import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    data_num, feature_num = x_train.shape
    alpha = torch.zeros(data_num)
    for a in range(num_iters):
        gradient = torch.zeros(data_num)
        for i in range(data_num):
            for j in range(data_num):
                gradient[i] += alpha[j] * y_train[i] * y_train[j] * kernel(x_train[i],x_train[j])
        gradient -= 1
        alpha -= lr * gradient
        alpha = torch.clamp(alpha,0,c)
    return alpha

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    data_num, feature_num = x_test.shape
    output = torch.zeros(data_num)
    for i in range(data_num):
        for k in range(x_train.shape[0]):
            output[i] += alpha[k] * y_train[k] * kernel(x_train[k],x_test[i])
    return output

x_train, y_train = hw2_utils.xor_data()
lr = 0.1
num_iters = 10000
kernel = hw2_utils.rbf(4)
alpha = svm_solver(x_train,y_train,lr,num_iters,kernel)
hw2_utils.svm_contour(lambda x_test : svm_predictor(alpha,x_train,y_train,x_test,kernel))


