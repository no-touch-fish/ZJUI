import numpy as np

def estimate_geometric(PX):
    '''
    @param:
    PX (numpy array of length cX): PX[x] = P(X=x), the observed probability mass function

    @return:
    p (scalar): the parameter of a matching geometric random variable
    PY (numpy array of length cX): PY[x] = P(Y=y), the first cX values of the pmf of a
      geometric random variable such that E[Y]=E[X].
    '''
    # raise RuntimeError("You need to write this")
    # get the mean of x
    x0 = len(PX)
    mean_x = 0
    for x in range(x0):
        mean_x += x * PX[x]
    # get the p
    p = 1 / (1 + mean_x)
    # get the PY
    PY = np.zeros(x0)
    for x in range(x0):
        PY[x] = p * (1- p)**x
    return p, PY
