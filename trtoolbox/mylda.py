import numpy as np
from scipy.linalg import svd
# from scipy.signal import medfilt
# from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class Results:

    def __init__(self):
        self.data = np.array([])
        self.time = np.array([])
        self.wn = np.array([])
        self.taus = np.array([])
        self.dmatrix = np.array([])
        self.alphas = np.array([])
        self.lmatrix = np.array([])
        self.k = None
        self.method = ''
        self.x_k = np.array([])
        self.wn_name = 'wavenumber'
        self.wn_unit = 'cm^{-1}'
        self.time_name = 'time'
        self.time_unit = 's'

    def plotlda(self, alpha=-1):
        plt.figure()
        if len(self.x_k.shape) == 3:
            if alpha == -1:
                alpha = int(np.ceil(self.alphas.size/2))
            x_k = self.x_k[:, :, alpha]
            title = 'LDA map at alpha = %f' % (self.alphas[alpha])
        else:
            x_k = self.x_k
            title = 'LDA map using TSVD'
        plt.pcolormesh(
            np.transpose(self.taus),
            self.wn,
            np.transpose(x_k),
            cmap='jet',
            shading='gouraud')
        plt.xscale('log')
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))
        plt.title(title)


def gen_taus(t1, t2, n):
    taus = np.logspace(np.log10(t1), np.log10(t2), n)
    return taus


def gen_dmatrix(time, taus):
    dmatrix = np.zeros([time.size, taus.size])
    for i in range(len(taus)):
        dmatrix[:, i] = (np.exp(-time/taus[i])).reshape(-1)
    return dmatrix


def gen_lmatrix(dmatrix):
    lmatrix = np.identity(np.shape(dmatrix)[1])
    b = np.ones(np.shape(dmatrix)[1])
    np.fill_diagonal(lmatrix[:, 1:], -b)
    return lmatrix


def gen_alphas(a1, a2, n):
    # alphas = np.linspace(a1, a2, n)
    alphas = np.logspace(np.log10(a1), np.log10(a2), n)

    if a1 > 1e-2:
        alphas = np.insert(alphas, 0, [1e-5, 1e-4, 1e-3, 1e-2])
    if a2 < 10:
        alphas = np.append(alphas, [10, 40, 70, 100])
    return alphas


def inversesvd(dmatrix, k=0):
    u, s, vt = svd(dmatrix, full_matrices=False)

    if k == 0:
        k = len(s)

    s = 1/s
    sigma = np.array([s[i] if i < k else 0 for i in range(len(s))])
    sigma = np.diag(sigma)

    ut = np.transpose(u)
    v = np.transpose(vt)

    return v.dot(sigma).dot(ut)


def tik(data, dmatrix, alpha):
    lamtrix = gen_lmatrix(dmatrix)

    if alpha != 0:
        d_aug = np.concatenate((dmatrix, alpha**(2)*lamtrix))
        a_aug = np.concatenate(
            (data, np.zeros([np.shape(data)[0], len(lamtrix)])),
            axis=1)
    else:
        d_aug = dmatrix
        a_aug = data

    d_tilde = inversesvd(d_aug)
    x_k = d_tilde.dot(np.transpose(a_aug))
    return x_k


def tiks(data, dmatrix, a1, a2, n):
    alphas = gen_alphas(a1, a2, n)

    x_ks = np.empty([np.shape(dmatrix)[1], np.shape(data)[0], len(alphas)])
    for i, alpha in enumerate(alphas):
        x_k = tik(data, dmatrix, alpha)
        x_ks[:, :, i] = x_k

    return x_ks


def calc_lcurve(data, dmatrix, lmatrix, x_ks):
    lcurve = np.empty((np.shape(x_ks)[2], 2))
    for i in range(np.shape(x_ks)[2]):
        lcurve[i, 0] = np.sum(
            (dmatrix.dot(x_ks[:, :, i])-np.transpose(data))**2)**(0.5)
        lcurve[i, 1] = np.sum((lmatrix.dot(x_ks[:, :, i]))**2)**(0.5)
    return lcurve


def tik_lstsq(data, dmatrix, alpha):
    lmatrix = gen_lmatrix(dmatrix)

    if alpha != 0:
        d_aug = np.concatenate((dmatrix, alpha**(2)*lmatrix))
        a_aug = np.concatenate(
            (data, np.zeros([np.shape(data)[0], len(lmatrix)])),
            axis=1)
    else:
        d_aug = dmatrix
        a_aug = data

    res = np.linalg.lstsq(d_aug, np.transpose(a_aug), rcond=None)
    return res


def tsvd(data, dmatrix, k):
    d_tilde = inversesvd(dmatrix, k)
    x_k = d_tilde.dot(np.transpose(data))
    return x_k


def dolda(
        data,
        time,
        wn,
        tlimits=[],
        tnum=100,
        alimits=[0.1, 5],
        anum=100,
        method='tik',
        k=5,
        prompt='no'):

    if prompt == 'no':
        if not tlimits:
            tlimits = [time[0], time[-1]]
    elif prompt == 'yes':
        method = input('Which method (tik or tsvd)? ')
        if method == tik:
            t1 = int(input('Bottom limit for time constants: '))
            t2 = int(input('Upper limit for time constants: '))
            tlimits = [t1, t2]
            tnum = int(input('Number of time constants: '))
            a1 = int(input('Bottom limit for alpha values: '))
            a2 = int(input('Upper limit for alpha values: '))
            alimits = [a1, a2]
            anum = int(input('Number of alpha values'))
        elif method == 'tsvd':
            k = int(input('How many singular values? '))

    taus = gen_taus(tlimits[0], tlimits[1], tnum)
    dmatrix = gen_dmatrix(time, taus)

    res = Results()
    res.data = data
    res.time = time
    res.wn = wn
    res.taus = taus
    res.dmatrix = dmatrix
    res.method = method
    if method == 'tik':
        res.alphas = gen_alphas(alimits[0], alimits[1], anum)
        x_k = tiks(data, dmatrix, alimits[0], alimits[1], anum)
    elif method == 'tsvd':
        res.k = k
        x_k = tsvd(data, dmatrix, k)

    res.x_k = x_k
    return res


#############################################
# Functions for lcruve curverature
# (currently not working!)
#############################################

# def calc_k(lcurve, alphas):
#     a = alphas[4:-4]
#     x = medfilt(np.log10(lcurve[3:-3,0]))
#     y = medfilt(np.log10(lcurve[3:-3,1]))
#     x = x[1:-1]
#     y = y[1:-1]
#     x_new = np.linspace(x[1], x[-1], 1000)
#     #da = np.gradient(a)
#     da = np.arange(x_new.size)
#     f = interp1d(x, y, kind='cubic')
#     plt.figure(); plt.plot(x_new, f(x_new))
#     dx = np.gradient(x_new)
#     dy = np.gradient(f(x_new))
#     dx2 = np.gradient(dx)
#     dy2 = np.gradient(dy)
#     k = 2*(dx*dy2 - dx2*dy) / (dx**2 + dy**2)**(1.5)
#     return k

# def calc_k_angle(lcurve, alphas):
#     #x = medfilt(np.log10(lcurve[:,0]), 7)
#     x = np.log(lcurve[:,0])
#     y = np.log(lcurve[:,1])
#     f = interp1d(x[0::10], y[0::10], kind='cubic')
#     npoints = 100
#     x_espaced = np.linspace(x[0], x[-10], npoints)
#     xy = np.transpose([x_espaced, f(x_espaced)])

#     plt.figure(); plt.plot(xy[:,0], xy[:,1])

#     angle = np.zeros([npoints, 1])
#     diff = np.zeros([npoints, 1])
#     max_diff = 0
#     knee = 0
#     for i in range(1,npoints-1):
#         v = xy[i,:]-xy[i-1,:]
#         w = xy[i+1,:]-xy[i,:]
#         angle[i] = np.arccos(v.dot(w)/(np.linalg.norm(v)*np.linalg.norm(w)))

#         a = angle[i-1]
#         a1 = angle[i-2]
#         d1 = a1 - a
#         a2 = angle[i]
#         d2 = a2 - a

#         diff[i] = d1 + d2
#         if diff[i] > max_diff:
#             max_diff = diff[i]
#             knee = lcurve[i-1,:]

#     return angle, diff, knee
