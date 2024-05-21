'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    # raise RuntimeError("You need to write this part!")
    word_list = []
    # loop to find the word count
    for text in texts:
        count = 0
        for word in text:
            if word == word0:
                count += 1 
        word_list.append(count)
    # create the pmarginal
    max_count = max(word_list)
    # print("word list is ")
    # print(word_list)
    Pmarginal = np.zeros(max_count+1)
    # put word list to Pmarginal
    for i in range(len(word_list)):
        Pmarginal[word_list[i]] += 1
    sum = 0
    for i in range(len(Pmarginal)):
        sum += Pmarginal[i]
    # change to P(x0 = x0)
    # print("before divide")
    # print(Pmarginal)
    Pmarginal = Pmarginal / sum
    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    # raise RuntimeError("You need to write this part!")
    # get the shape of the Pcond
    x0 = x1 = 0
    for text in texts:
      cx0 = cx1 = 0
      for word in text:
          if word == word0:
              cx0 += 1
          if word == word1:
              cx1 += 1
          x0 = max (x0,cx0)
          x1 = max (x1,cx1)
    Pjoint = np.zeros((x0 + 1,x1 + 1))
    Pcond = np.zeros((x0 + 1,x1 + 1))
    # count the times
    for text in texts:
        cx0 = cx1 = 0
        for word in text:
          if word == word0:
              cx0 += 1
          if word == word1:
              cx1 += 1
        Pjoint[cx0][cx1] += 1
    # get the probability of Pjoint
    Pjoint = Pjoint / len(texts)
    Pmarginal = marginal_distribution_of_word_counts(texts, word0)
    # get the Pcond
    for i in range(x0 + 1):
        for j in range(x1 + 1):
            if Pmarginal[i] != 0:
                Pcond[i][j] = Pjoint[i][j] / Pmarginal[i]
            else:
                Pcond[i][j] = np.nan
    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''
    # raise RuntimeError("You need to write this part!")
    Pjoint = np.zeros_like(Pcond)
    cx0,cx1 = Pcond.shape
    for x0 in range(cx0):
        # zero part
        if Pmarginal[x0] == 0:
            for x1 in range(cx1):
              Pjoint[x0][x1] = 0
            continue  
        for x1 in range(cx1):
            # zero part
            if Pcond[x0][x1] == np.nan:
                Pjoint[x0][x1] = 0
            # normal
            Pjoint[x0][x1] = Pmarginal[x0] * Pcond[x0][x1]
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    # raise RuntimeError("You need to write this part!")
    mu = np.zeros(2)
    mean_x0 = 0
    mean_x1 = 0
    x0,x1 = Pjoint.shape
    # mean of x0
    sum_x0 = np.sum(Pjoint,axis=1)
    for x in range(x0):
        mean_x0 += x * sum_x0[x]
    # mean of x1
    sum_x1 = np.sum(Pjoint,axis=0)
    for x in range(x1):
        mean_x1 += x * sum_x1[x]
    
    mu[0] = mean_x0
    mu[1] = mean_x1
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    # raise RuntimeError("You need to write this part!")
    Sigma = np.zeros((2,2))
    x0,x1 = Pjoint.shape
    # variance x0
    var_x0 = 0
    mean_x0 = mu[0]
    sum_x0 = np.sum(Pjoint,axis=1)
    for x in range(x0):
        var_x0 += sum_x0[x] * (x - mean_x0)**2
    var_x0 = var_x0
    # variance x1
    var_x1 = 0
    mean_x1 = mu[1]
    sum_x1 = np.sum(Pjoint,axis=0)
    for y in range(x1):
        var_x1 += sum_x1[y] * (y - mean_x1)**2
    var_x1 = var_x1
    # covirance of x0,x1
    mean_x0x1 = 0
    cov_x0x1 = 0
    for x in range(x0):
        for y in range(x1):
            mean_x0x1 += Pjoint[x][y] * x * y
    cov_x0x1 = mean_x0x1 - mean_x0 * mean_x1
    # get Sigma
    Sigma[0][0] = var_x0
    Sigma[1][1] = var_x1
    Sigma[0][1] = cov_x0x1
    Sigma[1][0] = cov_x0x1
    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    #raise RuntimeError("You need to write this part!")
    x0,x1 = Pjoint.shape
    # create the counter
    Pfunc = Counter()
    # add to the counter
    for x in range(x0):
        for y in range(x1):
            z = f(x,y)
            Pfunc[z] += Pjoint[x][y]
    return Pfunc
    
