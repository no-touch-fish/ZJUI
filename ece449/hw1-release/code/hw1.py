import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt

def linear_gd(X,Y,lrate=0.1,num_iter=1000):
    # return parameters as numpy array
    # get the shape
    num_data,feature = np.shape(X)
    ones_column = np.ones(num_data)
    ones_column = np.reshape(ones_column,(-1,1))
    X = np.hstack((ones_column,X))
    w = np.zeros(feature + 1)
    # loop
    for i in range(num_iter):
        # add the gradient
        gradient = np.zeros(feature + 1)
        for j in range(num_data):
            gradient += np.dot((np.dot(w.T,X[j]) - Y[j]),X[j])
        w = w - lrate * gradient * (1 / num_data)
    return w

def linear_normal(X,Y):
    # return parameters as numpy array
    num_data = X.shape[0]
    ones_column = np.ones(num_data)
    ones_column = np.reshape(ones_column,(-1,1))
    X = np.hstack((ones_column,X))
    matrix_1 = np.linalg.pinv(np.dot(X.T,X))
    matrix_2 = np.dot(matrix_1,X.T)
    w = np.dot(matrix_2,Y)
    return w

def plot_linear():
    # return plot
    X,Y = utils.load_reg_data()
    w = linear_normal(X,Y)
    plt.scatter(X,Y,color = 'blue',label = 'data point')
    x_value = np.linspace(0,4,num=100)
    y_value = w[0] + w[1] * x_value
    plt.plot(x_value,y_value,color = 'red',label = 'Linear Regression')
    plt.legend()
    plt.show()
    return []

plot_linear()