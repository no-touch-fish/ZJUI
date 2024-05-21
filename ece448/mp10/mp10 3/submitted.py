'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    M = model.M
    N = model.N
    P = np.zeros((M,N,4,M,N))
    for r in range(M):
        for c in range(N):
            if model.TS[r,c] == True:
                continue
            for a in range(4):
                    movements = []
                    if a == 0:  # left
                        movements = [(0, -1), (1, 0), (-1, 0)]
                    elif a == 1:  # up
                        movements = [(-1, 0), (0, -1), (0, 1)]
                    elif a == 2:  # right
                        movements = [(0, 1), (-1, 0), (1, 0)]
                    elif a == 3:  # down
                        movements = [(1, 0), (0, 1), (0, -1)]
                    for idx, (dr, dc) in enumerate(movements):
                        r_next = r + dr
                        c_next = c + dc
                        # Boundary check
                        Bound = r_next < 0 or r_next >= model.M or c_next < 0 or c_next >= model.N 
                        if Bound or model.W[r_next, c_next]:
                            r_next, c_next = r, c
                        P[r, c, a, r_next, c_next] += model.D[r, c, idx]
    return P

def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    M,N = model.M, model.N
    U_next = np.zeros((M, N))

    for r in range(M):
        for c in range(N):

            best_value = -np.inf
            for a in range(4):
                value = 0
                for r_next in range(M):
                    for c_next in range(N):
                        value += P[r, c, a, r_next, c_next] * U_current[r_next, c_next]

                best_value = max(best_value, value)

            U_next[r, c] = model.R[r, c] + model.gamma * best_value

    return U_next    

def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    # Compute the transition matrix
    P = compute_transition(model)

    # Initialize the utility function
    M,N = model.M, model.N
    U_current = np.zeros((M, N))
    # 100 is the max iteration
    for _ in range(100):
        # Update the utility function
        U_next = compute_utility(model, U_current, P)
        # Check for convergence
        if np.all(np.abs(U_next - U_current) < epsilon):
            break
        # Update U_current for the next iteration
        U_current = U_next
    return U_current
    

def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''
    M,N = model.M, model.N
    U = np.zeros((M, N))
    for _ in range(500):
        U_tmp = U.copy()
        for r in range(M):
            for c in range(N):
                value = 0
                for r_next in range(M):
                    for c_next in range(N):
                        value += model.FP[r, c, r_next, c_next] * U_tmp[r_next, c_next]
                U[r, c] = model.R[r, c] + model.gamma * value
        if np.all(np.abs(U - U_tmp) < epsilon):
            return U
