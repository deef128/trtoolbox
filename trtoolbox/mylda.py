# TODO: species associated spectra
import numpy as np
from scipy.linalg import svd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from trtoolbox.plothelper import PlotHelper


class Results:
    """ Object containing fit results.

    Attributes
    ----------
    type : str
        Results object type.
    data : np.array
        Data matrix subjected to fitting.
    time : np.array
        Time array
    wn : np.array
        Frequency array.
    taus: np.array
        Time constants used for constructing D-matrix.
    dmatrix : np.array
        D-matrix.
    alphas : np.array
        Used alpha values for computation (tik method).
    lmatrix : np.array
        L-matrix (tik method).
    lcurve : np.array
        L-curve.
    k : int
        Used SVD Components (tsvd method).
    method : string
        Used method (tik or tsvd).
    x_k : np.array
        Resulting exponential pre-factors.
    fitdata : np.array
        Constructed data (dmatrix.dot(x_k))
    wn_name : str
        Name of the frequency unit (default: wavenumber).
    wn_unit : str
        Frequency unit (default cm^{-1}).
    t_name : str
        Time name (default: time).
    time_uni : str
        Time uni (default: s).
    """

    def __init__(self):
        self.type = 'lda'
        self.data = np.array([])
        self.time = np.array([])
        self.wn = np.array([])
        self.taus = np.array([])
        self.dmatrix = np.array([])
        self.alphas = np.array([])
        self.lmatrix = np.array([])
        self.lcurve = np.array([])
        self.k = None
        self.method = ''
        self.x_k = np.array([])
        self.fitdata = np.array([])
        self.wn_name = 'wavenumber'
        self.wn_unit = 'cm^{-1}'
        self.time_name = 'time'
        self.time_unit = 's'
        self._phelper = PlotHelper()

    def init_phelper(self):
        if type(self._phelper) == list:
            self._phelper = PlotHelper()

    def get_alpha(self, index_alpha=-1, alpha=-1):
        """ Gets alpha value and index.

        Parameters
        ----------
        index_alpha : int
            Plot for specified alpha at index.
        alpha : float
            Plot for the closest alpha as specified.

        Returns
        -------
        index_alpha : int
            Index of alpha value.
        alpha : float
            Alpha value.
        """

        if index_alpha == -1 and alpha == -1:
            # if no alpha is specified take the middle of the alpha array
            index_alpha = int(np.ceil(self.alphas.size/2))
        elif alpha != -1:
            # search for closest alpha value
            index_alpha = (np.abs(self.alphas - alpha)).argmin()
        alpha = self.alphas[index_alpha]
        return index_alpha, alpha

    def get_xk(self, index_alpha=-1, alpha=-1):
        """ Gets selected LDA map.

        Parameters
        ----------
        index_alpha : int
            Plot for specified alpha at index.
        alpha : float
            Plot for the closest alpha as specified.

        Returns
        -------
        x_k : np.array
            LDA map.
        title : str
            Title for figure.
        """

        # check for used method
        if len(self.x_k.shape) == 3:
            index_alpha, _ = self.get_alpha(index_alpha, alpha)
            x_k = self.x_k[:, :, index_alpha]
            title = 'LDA map at alpha = %f' % (self.alphas[index_alpha])
        else:
            x_k = self.x_k
            title = 'LDA map using TSVD'

        return x_k, title

    def plot_ldamap(self, index_alpha=-1, alpha=-1):
        """ Plots a nice looking contourmap.

        Parameters
        ----------
        index_alpha : int
            Plot for specified alpha at index.
        alpha : float
            Plot for the closest alpha as specified.

        Returns
        -------
        nothing
        """

        self.init_phelper()
        if self.x_k.size == 0:
            print('First start a LDA.')
            return
        x_k, title = self.get_xk(index_alpha, alpha)

        plt.figure()
        self._phelper.plot_contourmap(
            x_k, self.taus, self.wn,
            title=title, newfig=True)
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % ('tau', self.time_unit))
        plt.title(title)

    def plot_traces(self, index_alpha=-1, alpha=-1):
        """ Plots interactive time traces.

        Returns
        -------
        nothing
        """

        self.init_phelper()
        self._phelper.plot_traces(self, index_alpha, alpha)

    def plot_spectra(self, index_alpha=-1, alpha=-1):
        """ Plots interactive spectra.

        Returns
        -------
        nothing
        """

        self.init_phelper()
        self._phelper.plot_spectra(self, index_alpha, alpha)

    def plot_lcurve(self):
        """ Plots L-curve.

        Returns
        -------
        nothing
        """

        plt.figure()
        plt.plot(self.lcurve[:, 0], self.lcurve[:, 1], 'o-', markersize=2)

    def plot_solutionvector(self, index_alpha=-1, alpha=-1):
        """ Plots the sum of amplituted over time constants.

        Returns
        -------
        nothing
        """

        plt.figure()
        x_k, title = self.get_xk(index_alpha, alpha)
        # plt.plot(
        #     [np.min(self.taus), np.max(self.taus)],
        #     [0, 0],
        #     '--', color='k'
        # )
        plt.plot(self.taus.T, np.sum(np.abs(x_k), axis=0))
        plt.xscale('log')
        time_min = np.min(self.time[0, :])
        time_max = np.max(self.time[0, :])
        plt.xlim([time_min, time_max])
        plt.xlabel('time constant / ' + self.time_unit)
        plt.title('Solution vector ' + title[8:])

    def plot_results(self):
        """ Plots interactive contourmaps of original and LDA data,

        Returns
        -------
        nothing
        """

        self.init_phelper()
        self._phelper.plot_ldaresults(self)

    def clean(self):
        """ Unfortunetaly, spyder messes up when the results
            object is invesitgated via the variable explorer.
            Running this method fixes this.

        Returns
        -------
        nothing
        """

        self._phelper = []


def check_input(data, time, wn):
    """ Ensures that all np.arrays have float dtype and that
        time spans over columns, frequency over rows.

    Parameters
    ----------
    data : np.array
        Data matrix.
    time : np.array
        TIme array.
    wn : np.array
        Frequency array.

    Returns
    -------
    nothing
    """

    # check for right dtype
    if data.dtype != 'float':
        data = data.astype('float64')
    if time.dtype != 'float':
        time = time.astype('float64')
    if wn.dtype != 'float':
        wn = wn.astype('float64')

    # ensure time over columns and
    # frequency over rows
    if data.shape[1] != time.size:
        data = np.transpose(data)
    time = time.reshape((1, time.size))
    wn = wn.reshape((wn.size, 1))

    return data, time, wn


def gen_taus(t1, t2, n):
    """ Generates logarihmic spaced time constants.

    Parameters
    ----------
    t1 : float
        Bottom limit.
    t2 : float
        Upper limit.
    n : int
        Number of time constants.

    Returns
    -------
    taus : np.array
        Generated time constants.
    """

    taus = np.logspace(np.log10(t1), np.log10(t2), n)
    return taus


def gen_dmatrix(time, taus, seqmodel=False):
    """ Generates D-matrix.

    Parameters
    ----------
    time : np.array
        Time array.
    taus : np.array
        Time constants array

    Returns
    -------
    dmatrix : np.array
        D-matrix
    """
    if seqmodel is True:
        def model(s, time, ks):
            arr = [-ks[0] * s[0]]
            arr.append(ks[0] * s[0] - ks[1] * s[1])
            return arr

    dmatrix = np.zeros([time.size, taus.size])
    for i in range(len(taus)):
        if i > 0 and seqmodel is True:
            # dmatrix[:, i] = \
            #     (-1*dmatrix[:, i-1] + np.exp(-time/taus[i])).reshape(-1)
            ks = 1./taus[np.array([i-1, i])]
            res = odeint(model, [1, 0], time.flatten(), (ks,))
            dmatrix[:, i] = res[:, 1]
        else:
            dmatrix[:, i] = (np.exp(-time/taus[i])).reshape(-1)
    return dmatrix


def gen_lmatrix(dmatrix):
    """ Generates L-matrix

    Parameters
    ----------
    damtrix : np.array
        D-matrix

    Returns
    -------
    lamtrix : np.array
        L-matrix
    """

    lmatrix = np.identity(np.shape(dmatrix)[1])
    b = np.ones(np.shape(dmatrix)[1])
    np.fill_diagonal(lmatrix[:, 1:], -b)
    return lmatrix


def gen_alphas(a1, a2, n, space='log'):
    """ Generates logarihmic spaced alpha values.
    Adds [1e-5, 1e-4, 1e-3, 1e-2] and [10, 40, 70, 100]
    for a better visualization of the L-curve.

    Parameters
    ----------
    a1 : float
        Bottom limit.
    a2 : float
        Upper limit.
    n : int
        Number of alpha values.
    space : str
        *log* for logarithmic and *lin* for linear spaced

    Returns
    -------
    alphas : np.array
        Generated alpha values.
    """

    if space == 'log':
        alphas = np.logspace(np.log10(a1), np.log10(a2), n)
    elif space == 'lin':
        alphas = np.linspace(a1, a2, n)

    # code snippet to append alpha values for a better lcurce representation
    # if a1 > 1e-2:
    #     alphas = np.insert(alphas, 0, [1e-5, 1e-4, 1e-3, 1e-2])
    # if a2 < 10:
    #     alphas = np.append(alphas, [10, 40, 50, 100])
    return alphas


def inversesvd(dmatrix, k=-1):
    """ Returns the inverse of matrix computed via SVD.

    Parameters
    ----------
    dmatrix : np.array
        Matrix to be inversed

    Returns
    -------
     : np.array
        Inverse of input matrix.
    """

    u, s, vt = svd(dmatrix, full_matrices=False)

    if k == -1:
        k = len(s)

    s = 1/s
    sigma = np.array([s[i] if i < k else 0 for i in range(len(s))])
    sigma = np.diag(sigma)

    ut = np.transpose(u)
    v = np.transpose(vt)

    return v.dot(sigma).dot(ut)


# TODO: option for truncation
def tik(data, dmatrix, alpha):
    """ Function for Tikhonov regularization:
        min_x ||Dx - A|| + alpha*||Lx||
        D-matrix contains exponential profiles,
        x are prefactors/amplitudes,
        A is the dataset,
        alpha is the regularization factor and
        L is the identity matrix.

        Details can be found in
        Dorlhiac, Gabriel F. et al.
        "PyLDM-An open source package for lifetime density analysis
        of time-resolved spectroscopic data."
        PLoS computational biology 13.5 (2017)

    Parameters
    ----------
    data : np.array
        Data matrix to be analyzed
    dmatrix : np.array
        D-matrix
    alpha : float
        Regularization factor

    Returns
    -------
    x_k : np.array
        Expontential prefactors/amplitudes.
    """

    lmatrix = gen_lmatrix(dmatrix)

    # constructing augmented D- and A-matrices.
    # d_aug = (D, sqrt(alpha)*L)
    # a_aug = (A, zeros)
    if alpha != 0:
        d_aug = np.concatenate((dmatrix, alpha**(0.5)*lmatrix))
        a_aug = np.concatenate(
            (data, np.zeros([np.shape(data)[0], len(lmatrix)])),
            axis=1)
    else:
        d_aug = dmatrix
        a_aug = data

    d_tilde = inversesvd(d_aug)
    x_k = d_tilde.dot(np.transpose(a_aug))
    return x_k


def tiks(data, dmatrix, alphas):
    """ Wrapper for computing LDA for
        various alpha values.

    Parameters
    ----------
    data : np.array
        Data matrix to be analyzed.
    dmatrix : np.array
        D-matrix.
    alphas : np.array
        Array of regularization factors.

    Returns
    -------
    x_k : np.array
        3D matrix of expontential prefactors/amplitudes.
    """

    x_ks = np.empty([np.shape(dmatrix)[1], np.shape(data)[0], len(alphas)])
    for i, alpha in enumerate(alphas):
        x_k = tik(data, dmatrix, alpha)
        x_ks[:, :, i] = x_k

    return x_ks


def calc_lcurve(data, dmatrix, lmatrix, x_ks):
    """ Calculates L-curve.

    Parameters
    ----------
    data : np.array
        Data matrix.
    dmatrix : np.array
        D-matrix.
    lmatrix : np.array
        L-matrix.
    x_ks : np.array
        LDA maps.

    Returns
    -------
    lcurve : np.array
        First column is resdiual norm.
        Second column is smoothed norm.
    """

    lcurve = np.empty((np.shape(x_ks)[2], 2))
    for i in range(np.shape(x_ks)[2]):
        lcurve[i, 0] = np.sum(
            (dmatrix.dot(x_ks[:, :, i])-np.transpose(data))**2)**(0.5)
        lcurve[i, 1] = np.sum((lmatrix.dot(x_ks[:, :, i]))**2)**(0.5)
    return lcurve


def tik_lstsq(data, dmatrix, alpha):
    """ Different implementation of the tik function.
        Uses the ordinary lstsq solver of numpy.

    Parameters
    ----------
    data : np.array
        Data matrix to be analyzed.
    dmatrix : np.array
        D-matrix.
    alpha : float
        Regularization factor.

    Returns
    -------
    res : np.array
        Expontential prefactors/amplitudes.
    """
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
    """ Truncated SVD for LDA. Similar to Tikhonov regularization
        but here we have a clear cut-off after a specified singular value.

        Details can be found in
        Hansen PC.
        The truncated SVD as a method for regularization.
        Bit. 1987

    Parameters
    ----------
    data : np.array
        Data matrix to be analyzed.
    dmatrix : np.array
        D-matrix.
    k : int
        Cut-off for singular values.

    Returns
    -------
    x_k : np.array
        Expontential prefactors/amplitudes.
    """

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
        seqmodel=False,
        k=5,
        prompt=False):
    """ Wrapper for doing a LDA.

    Parameters
    ----------
    data : np.array
        Data matrix subjected to SVD. Assuming *m x n* with m as frequency
        and n as time. But it is actually not important.
    time : np.array
        Time array.
    wn : np.array
        Frequency array.
    tlimits : list
        Limits for time constants.
    tnum : int
        Number of time constants.
    alimits : list
        Limits for alpha values.
    anum : int
        Number if alpha values.
    method : str
        Chosen method for LDA. Either 'tik' or 'tsvd'.
    seqmodel : boolean
        True for constructing the D-matrix assuming a sequential model.
    k : int
        Just used for 'tsvd'. Specifies the position of truncation.
    prompt : boolean
        True for user prompts.

    Returns
    -------
    res : *mylda.results()*
        Results object.
    """

    data, time, wn = check_input(data, time, wn)

    if prompt is False:
        if not tlimits:
            tlimits = [time[0, 0], time[0, -1]]
    elif prompt is True:
        method = input('Which method (tik or tsvd)? ')
        if method == 'tik':
            t1 = float(input('Bottom limit for time constants: '))
            t2 = float(input('Upper limit for time constants: '))
            tlimits = [t1, t2]
            tnum = int(input('Number of time constants: '))
            a1 = float(input('Bottom limit for alpha values: '))
            a2 = float(input('Upper limit for alpha values: '))
            alimits = [a1, a2]
            anum = int(input('Number of alpha values: '))
        elif method == 'tsvd':
            k = int(input('How many singular values? '))

    taus = gen_taus(tlimits[0], tlimits[1], tnum)
    dmatrix = gen_dmatrix(time, taus, seqmodel=seqmodel)
    lmatrix = gen_lmatrix(dmatrix)

    res = Results()
    res.data = data
    res.time = time
    res.wn = wn
    res.taus = taus.reshape((1, taus.size))
    res.dmatrix = dmatrix
    res.method = method
    if method == 'tik':
        res.alphas = gen_alphas(alimits[0], alimits[1], anum)
        x_k = tiks(data, dmatrix, res.alphas)
        res.lmatrix = gen_lmatrix(dmatrix)
        res.lcurve = calc_lcurve(data, dmatrix, res.lmatrix, x_k)
        fitdata = np.empty(data.shape + (np.shape(x_k)[2], ))
        for i in range(np.shape(x_k)[2]):
            fitdata[:, :, i] = np.transpose(dmatrix.dot(x_k[:, :, i]))
        res.fitdata = fitdata
        res.lcurve = calc_lcurve(data, dmatrix, lmatrix, x_k)
    elif method == 'tsvd':
        res.k = k
        x_k = tsvd(data, dmatrix, k)
        res.fitdata = np.transpose(dmatrix.dot(x_k))

    res.x_k = np.swapaxes(x_k, 0, 1)
    return res


#############################################
# Functions for lcurve curverature
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
