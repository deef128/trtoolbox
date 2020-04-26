import numpy as np
from scipy.linalg import svd

def gentaus(t1, t2, n):
    taus = np.logspace(np.log10(t1), np.log10(t2), n)
    return taus

def genD(time, taus):
    D = np.zeros([np.shape(time)[1], len(taus)])
    for i in range(len(taus)):
        D[:,i] = (np.exp(-time/taus[i])).reshape(-1)
    return D

def inversesvd(D, k):
    U, s, VT = svd(D, full_matrices=False)
    
    # create m x n singular values matrix
    # sigma = np.zeros((U.shape[0], VT.shape[0]))
    # sigma[:s.shape[0], :s.shape[0]] = np.diag(s)
    s = 1/s
    sigma = np.array([s[i] if i < k else 0 for i in range(len(s))])
    sigma = np.diag(sigma)
    
    UT = np.transpose(U)
    V = np.transpose(VT)
    
    return V.dot(sigma).dot(UT)
    
def tsvd(data, D, k):
    D_tilde = inversesvd(D, k)
    x_k = D_tilde.dot(np.transpose(data))
    return x_k



def genDd(time, taus):
    D = np.zeros([np.shape(time)[1], len(taus)])
    for i in range(len(D)):
        for j in range(len(D[i])):
            t = time[0,i]
            tau = taus[j]
            D[i, j] = np.exp(-t/tau)
    return D