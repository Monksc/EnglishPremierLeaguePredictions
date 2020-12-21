import numpy as np


def getD(A):
    return [[A[i].sum() if i == j else 0 for j in range(len(A))] for i in range(len(A))]

def findInverseOfD(D):
    return [[1 / D[i][j] if D[i][j] != 0 else 0 for j in range(len(D))] for i in range(len(D))]
    

def getM(A):
    D = getD(A)
    Dinv = findInverseOfD(D) # np.linalg.inv(D)
    return (Dinv @ A).T

# M = (D^-1 A)^T
# M is just prob that you move from row to collumn
# M should be ^inf power
def rank(A, power=2**30):
    M = getM(A)
    M = np.linalg.matrix_power(M, power)
    #SM = getM(A)
    #IM = SM
    #M = SM
    #while power != 1:
    #    if power % 2 == 0:
    #        power = power / 2
    #        IM = IM @ IM
    #    else:
    #        power -= 1
    #        M = M @ IM
    #        IM = SM
            
    P = np.reshape(np.array([1/len(A) for i in range(len(A))]), (len(A), 1))
    return M @ P


