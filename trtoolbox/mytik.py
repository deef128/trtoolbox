import numpy as np
from scipy.linalg import svd
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.optimize import minimize

import matplotlib.pyplot as plt

def gentaus(t1, t2, n):
    taus = np.logspace(np.log10(t1), np.log10(t2), n)
    return taus

def genD(time, taus):
    D = np.zeros([np.shape(time)[1], len(taus)])
    for i in range(len(taus)):
        D[:,i] = (np.exp(-time/taus[i])).reshape(-1)
    return D

def genL(D):
    L = np.identity(np.shape(D)[1])
    b = np.ones(np.shape(D)[1])
    np.fill_diagonal(L[:,1:], -b)
    return L

def genAlphas(a1, a2, n):
    #alphas = np.linspace(a1, a2, n)
    alphas = np.logspace(np.log10(a1), np.log10(a2), n)
    
    if a1 > 1e-2 :
        alphas = np.insert(alphas, 0, [1e-5, 1e-4, 1e-3, 1e-2])
    if a2 < 10 :
        alphas = np.append(alphas, [10, 40, 70, 100])
    return alphas

def inversesvd(D, k=0):
    U, s, VT = svd(D, full_matrices=False)
    
    if k == 0:
        k = len(s)
    
    # create m x n singular values matrix
    # sigma = np.zeros((U.shape[0], VT.shape[0]))
    # sigma[:s.shape[0], :s.shape[0]] = np.diag(s)
    s = 1/s
    sigma = np.array([s[i] if i < k else 0 for i in range(len(s))])
    sigma = np.diag(sigma)
    
    UT = np.transpose(U)
    V = np.transpose(VT)
    
    return V.dot(sigma).dot(UT)
    
def tik(data, D, alpha):
    L = genL(D)
    
    if alpha != 0:
        D_aug = np.concatenate((D, alpha**(2)*L))
        A_aug = np.concatenate((data, np.zeros( [np.shape(data)[0], len(L)] )), axis=1)
    else:
        D_aug = D
        A_aug = data
    
    D_tilde = inversesvd(D_aug)
    x_k = D_tilde.dot(np.transpose(A_aug))
    return x_k

def tiks(data, D, a1, a2, n):
    alphas = genAlphas(a1, a2, n)
    
    x_ks = np.empty( [np.shape(D)[1], np.shape(data)[0], len(alphas)] )
    for i, alpha in enumerate(alphas):
        x_k = tik(data, D, alpha)
        x_ks[:,:,i] = x_k
        
    return x_ks

def calc_lcurve(data, D, L, x_ks):
    lcurve = np.empty((np.shape(x_ks)[2], 2))
    for i in range(np.shape(x_ks)[2]):
        lcurve[i,0] = np.sum( (D.dot(x_ks[:,:,i])-np.transpose(data))**2 )**(0.5)
        lcurve[i,1] = np.sum( (L.dot(x_ks[:,:,i]))**2 )**(0.5)
    return lcurve

def tik_lstsq(data, D, alpha):
    L = genL(D)
    A = data
    x_ini = np.ones([D.shape[1], A.shape[0]])
    
    if alpha != 0:
        D_aug = np.concatenate((D, alpha**(2)*L))
        A_aug = np.concatenate((data, np.zeros( [np.shape(data)[0], len(L)] )), axis=1)
    else:
        D_aug = D
        A_aug = data
    
    #res = minimize(func_min, x_ini.flatten(), args=(D, A, L, alpha, D.shape[1]))
    res = np.linalg.lstsq(D_aug, np.transpose(A_aug), rcond=None)
    return res
        



#############################################
# Functions for lcruve curverature
# (currently not working!)
#############################################

def calc_k(lcurve, alphas):
    a = alphas[4:-4]
    x = medfilt(np.log10(lcurve[3:-3,0]))
    y = medfilt(np.log10(lcurve[3:-3,1]))
    x = x[1:-1]
    y = y[1:-1]
    x_new = np.linspace(x[1], x[-1], 1000)
    #da = np.gradient(a)
    da = np.arange(x_new.size)
    f = interp1d(x, y, kind='cubic')
    plt.figure(); plt.plot(x_new, f(x_new))
    dx = np.gradient(x_new)
    dy = np.gradient(f(x_new))
    dx2 = np.gradient(dx)
    dy2 = np.gradient(dy)
    k = 2*(dx*dy2 - dx2*dy) / (dx**2 + dy**2)**(1.5)
    return k

def calc_k_angle(lcurve, alphas):
    #x = medfilt(np.log10(lcurve[:,0]), 7)
    x = np.log(lcurve[:,0])
    y = np.log(lcurve[:,1])
    f = interp1d(x[0::10], y[0::10], kind='cubic')
    npoints = 100
    x_espaced = np.linspace(x[0], x[-10], npoints)
    xy = np.transpose([x_espaced, f(x_espaced)])
    
    plt.figure(); plt.plot(xy[:,0], xy[:,1])
    
    angle = np.zeros([npoints, 1])
    diff = np.zeros([npoints, 1])
    max_diff = 0
    knee = 0
    for i in range(1,npoints-1):
        v = xy[i,:]-xy[i-1,:]
        w = xy[i+1,:]-xy[i,:]
        angle[i] = np.arccos(v.dot(w)/(np.linalg.norm(v)*np.linalg.norm(w)))
        
        a = angle[i-1]
        a1 = angle[i-2]
        d1 = a1 - a
        a2 = angle[i]
        d2 = a2 - a
        
        diff[i] = d1 + d2
        if diff[i] > max_diff:
            max_diff = diff[i]
            knee = lcurve[i-1,:]
    
    return angle, diff, knee
            