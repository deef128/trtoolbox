import os
from scipy.linalg import svd as scipy_svd
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.pclasses as pclasses


class Results(pclasses.Data):
    """ Object containing fit results.

    Attributes
    ----------
    type : str
        Results object type.
    u : np.array
        U matrix. Represents abstract spectra
    s : np.array
        Singular values.
    vt: np.array
        Transposed V matrix. Represents abstract time traces.
    n : int
        Number of singular components used for data reconstruction.
    svddata : np.array
        Reconstructed data.
    """

    def __init__(self):
        super().__init__()
        self.type = 'svd'
        self.u = np.array([])
        self.s = np.array([])
        self.vt = np.array([])
        self.n = 0
        self.svddata = np.array([])

    # if using this rename self.wn to self._wn in init()
    # @property
    # def wn(self):
    #     return self._wn

    # @wn.setter
    # def wn(self, val):
    #     if val.dtype != 'float':
    #         val = val.astype('float64')
    #     self._wn = val.reshape((val.size, 1))

    def plot_data(self, newfig=True, interpolate=False, step=.5):
        """ Plots a nice looking heatmap of the raw data.

        Parameters
        ----------
        newfig : boolean
            True for own figure.
        interpolate : boolean
            True for interpolation
        step : float
            Step size for frequency interpolation.
        """

        self.init_phelper()
        self._phelper.plot_heatmap(
            self.data, self.time, self.wn,
            title='Original Data', newfig=newfig,
            interpolate=interpolate, step=step
        )
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_svddata(self, newfig=False, interpolate=False, step=.5):
        """ Plots a nice looking heatmap of the reconstructed data.
        """

        self.init_phelper()
        nstr = str(self.n)
        title = 'Reconstructed data using ' + nstr + ' components'
        self._phelper.plot_heatmap(
            self.svddata, self.time, self.wn,
            title=title, newfig=newfig,
            interpolate=interpolate, step=step
        )
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_svddata_3d(self, interpolate=False, step=.5):
        """ Plots 3D surface of the reconstructed data.

        Parameters
        ----------
        interpolate : boolean
            True for interpolation
        step : float
            Step size for frequency interpolation.
        """

        self.init_phelper()
        nstr = str(self.n)
        title = 'Reconstructed data using ' + nstr + ' components'
        self._phelper.plot_surface(
            self.svddata, self.time, self.wn,
            title=title,
            interpolate=interpolate, step=step
        )
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_traces(self):
        """ Plots interactive time traces.
        """

        self.init_phelper()
        self._phelper.plot_traces(self)

    def plot_spectra(self):
        """ Plots interactive spectra.
        """

        self.init_phelper()
        self._phelper.plot_spectra(self)

    def plot_abstract_traces(self):
        if type(self.n) == int:
            nlist = list(range(self.n))
        else:
            nlist = self.n

        cols = int(np.ceil(len(nlist)/2))
        fig, axs = plt.subplots(2, cols)
        fig.suptitle('Abstract traces')
        r = 0
        offset = 0
        for i in range(len(nlist)):
            if i == cols:
                r = 1
                offset = 4
            axs[r, i-offset].plot(self.time.T, self.vt[i, :])
            axs[r, i-offset].set_xscale('log')

    def plot_abstract_spectra(self):
        if type(self.n) == int:
            nlist = list(range(self.n))
        else:
            nlist = self.n

        cols = int(np.ceil(len(nlist)/2))
        fig, axs = plt.subplots(2, cols)
        fig.suptitle('Abstract spectra')
        r = 0
        offset = 0
        for i in range(len(nlist)):
            if i == cols:
                r = 1
                offset = 4
            axs[r, i-offset].plot(self.wn, self.u[:, i])

    def plot_results(self):
        """ Plots heatmaps of original and SVD data,
        interactive time traces and spectra.
        """

        self.init_phelper()
        # original data
        _, axs = plt.subplots(2, 1)
        plt.subplots_adjust(top=0.925)
        plt.subplots_adjust(bottom=0.075)
        plt.sca(axs[0])
        self._phelper.plot_heatmap(
            self.data, self.time, self.wn,
            title='Original Data', newfig=False)
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

        # svd data
        nstr = str(self.n)
        title = 'Reconstructed data using ' + nstr + ' components'
        plt.sca(axs[1])
        self._phelper.plot_heatmap(
            self.svddata, self.time, self.wn,
            title=title, newfig=False)
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

        self._phelper.plot_traces(self)
        self._phelper.plot_spectra(self)

    def save_to_files(self, path):
        """ Saving results to .dat files.

        Parameters
        ----------
        path : str
            Path for saving.
        """

        if os.path.exists(path) is False:
            answer = input('Path not found. Create (y/n)? ')
            if answer == 'y':
                os.mkdir(path)
            else:
                return

        to_save = ['data', 'svddata']
        for k, i in vars(self).items():
            if k in to_save:
                fname = k + '.dat'
                print('Writing ' + fname)
                np.savetxt(
                    os.path.join(path, fname),
                    i,
                    delimiter=',',
                    fmt='%.4e'
                )

        f = open(os.path.join(path, '00_comments.txt'), 'w')
        print('Writing 00_comments.txt')
        f.write('Created with trtoolbox\n' +
                '----------------------\n\n' +
                'SVD components used: %i\n' % self.n +
                '----------------------\n\n' +
                'Files:\n' +
                '\t- data.dat (Raw data)\n' +
                '\t- svddata.data (Reconstructed data)\n'
                )
        f.close()


def wrapper_svd(data):
    """ Simple wrapper for the *scipy.linalg.svd()* function.

    Parameters
    ----------
    data : np.array
        Data matrix subjected to SVD. Assuming *m x n* with m as frequency
        and n as time. But it is actually not important.

    Returns
    -------
    u : np.array
        U matrix. Represents abstract spectra
    s : np.array
        Singular values.
    vt: np.array
        Transposed V matrix. Represents abstract time traces.
    """

    # noinspection PyTupleAssignmentBalance
    u, s, vt = scipy_svd(data)
    return u, s, vt


def show_svs(data, time, wn):
    """ Plots singular values and variance explained.

    Parameters
    ----------
    data : np.array
        Data matrix subjected to SVD. Assuming *m x n* with m as frequency
        and n as time. But it is actually not important.
    time : np.array
        Time array.
    wn : np.array
        Frequency array.
    """

    data, time, wn = pclasses.check_input(data, time, wn)

    # noinspection PyTupleAssignmentBalance
    u, s, vt = scipy_svd(data)
    eig = s**2/np.sum(s**2)

    if s.size < 8:
        raise RuntimeError('Too less singular values!')
    if s.size < 15:
        num = s.size
    else:
        num = 15
    numlist = list(range(1, num+1))
    varlimits = [0.8, 0.95, 0.995]
    colors = ['red', 'orange', 'forestgreen']
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('First %i singular values' % num)
    axs[0].plot(numlist, s[:num], 'o-')
    axs[0].set_title('Singular values')
    axs[0].set_ylabel('|s|')
    axs[1].plot(numlist, np.cumsum(eig[:num])*100, 'o-')
    axs[1].set_title('Cummulative variance explained')
    axs[1].set_ylabel('variance explained / %')
    for i, limit in enumerate(varlimits):
        axs[1].plot(numlist, np.ones(num)*limit*100, '--', color=colors[i])
        svs = np.where(np.cumsum(eig) >= limit)[0][0]+1
        print(
            '%.1f %% variance explained by %i singular values'
            % (limit*100, svs))

    fig, axs = plt.subplots(2, 4)
    fig.suptitle('Abstract spectra')
    r = 0
    offset = 0
    for i in range(8):
        if i == 4:
            r = 1
            offset = 4
        axs[r, i-offset].plot(wn, u[:, i])

    fig, axs = plt.subplots(2, 4)
    fig.suptitle('Abstract time traces')
    r = 0
    offset = 0
    for i in range(8):
        if i == 4:
            r = 1
            offset = 4
        axs[r, i-offset].plot(time.T, vt[i, :])
        axs[r, i-offset].set_xscale('log')


def reconstruct(data, n):
    """ Reconstructs data with n singular components.

    Parameters
    ----------
    data : np.array
        Data matrix subjected to SVD. Assuming *m x n* with m as frequency
        and n as time. But it is actually not important.
    n : int, list or np.array
        Number of used SVD components.
        If a list or array is provided, non pythonic way of numbering is used.
        Meaning first component equals 1.

    Returns
    -------
    res : *mysvd.results*
        Results object.
    """

    # noinspection PyTupleAssignmentBalance
    u, s, vt = scipy_svd(data)
    nlist = []
    if type(n) == int:
        nlist = list(range(n))
    elif type(n) == list:
        n = np.array(n)
        nlist = n-1
    elif type(n) == np.ndarray:
        nlist = n-1

    if any(i < 0 for i in nlist) is True:
        raise ValueError('Please chose just positive singular values')
    if len(set(nlist)) != len(nlist):
        raise ValueError('Please choose different singular values.')

    # create m x n singular values matrix
    sigma = np.zeros((u.shape[0], vt.shape[0]))
    sigma[:s.shape[0], :s.shape[0]] = np.diag(s)

    # reconstruct data
    svddata = u[:, nlist].dot(sigma[nlist, :].dot(vt))

    res = Results()
    res.data = data
    res.u = u
    res.s = s
    res.vt = vt
    res.n = n
    res.svddata = svddata

    return res


def dosvd(data, time, wn, n=-1):
    """ Wrapper for inspecting SVD components with reconstruction.

    Parameters
    ----------
    data : np.array
        Data matrix subjected to SVD. Assuming *m x n* with m as frequency
        and n as time. But it is actually not important.
    time : np.array
        Time array.
    wn : np.array
        Frequency array.
    n : int, list or np.array
        Number of used SVD components.
        If a list or array is provided, non pythonic way of numbering is used.
        Meaning first component equals 1.

    Returns
    -------
    res : *svd.results*
        Results object.
    """

    data, time, wn = pclasses.check_input(data, time, wn)

    # prevents plt.show() from blocking execution
    # plt.ion()
    # input can also be a list like 1,2,3,5
    if type(n) == int and n <= 0:
        plt.ion()
        show_svs(data, time, wn)
        plt.show()
        plt.ioff()
        n = input('How many singular values? ')
        if n.find(',') == -1:
            n = int(n)
        elif n.find(',') > 0:
            if n[0] == '[' and n[-1] == ']':
                n = n[1:-1]
            n = [int(i) for i in n.split(',')]
        else:
            print('Wrong input')
            return
    res = reconstruct(data, n)
    res.time = time
    res.wn = wn
    return res
