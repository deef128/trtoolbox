# TODO: check for tranpose
from scipy.linalg import svd
import numpy as np
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
    wn : np.array
        Frequency array.
    wn_name : str
        Name of the frequency unit (default: wavenumber).
    wn_unit : str
        Frequency unit (default cm^{-1}).
    time : np.array
        Time array.
    t_name : str
        Time name (default: time).
    time_uni : str
        Time uni (default: s).
    __phelper : mysvd.PlotHelper
        Plot helper class for interactive plots.
    """

    def __init__(self):
        self.type = 'svd'
        self.data = np.array([])
        self.u = np.array([])
        self.s = np.array([])
        self.vt = np.array([])
        self.n = 0
        self.svddata = np.array([])
        self.time = np.array([])
        self.time_name = 'time'
        self.time_unit = 's'
        self.wn = np.array([])
        self.wn_name = 'wavenumber'
        self.wn_unit = 'cm^{-1}'
        self.__phelper = PlotHelper()

    def plotdata(self, newfig=True):
        """ Plots a nice looking heatmap of the raw data.

        Returns
        -------
        nothing
        """

        self.__phelper.plot_heatmap(
            self.data, self.time, self.wn,
            title='Original Data', newfig=False)
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plotsvddata(self):
        """ Plots a nice looking heatmap of the reconstructed data.

        Returns
        -------
        nothing
        """

        nstr = str(self.n)
        title = 'Reconstructed data using ' + nstr + ' components'
        self.__phelper.plot_heatmap(
            self.svddata, self.time, self.wn,
            title=title, newfig=False)
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_traces(self):
        """ Plots interactive time traces.

        Returns
        -------
        nothing
        """

        self.__phelper.plot_traces(self)

    def plot_spectra(self):
        """ Plots interactive spectra.

        Returns
        -------
        nothing
        """

        self.__phelper.plot_spectra(self)

    def plot_results(self):
        """ Plots heatmaps of original and SVD data,
        interactive time traces and spectra.

        Returns
        -------
        nothing
        """

        # original data
        _, axs = plt.subplots(2, 1)
        plt.subplots_adjust(top=0.925)
        plt.subplots_adjust(bottom=0.075)
        plt.sca(axs[0])
        self.__phelper.plot_heatmap(
            self.data, self.time, self.wn,
            title='Original Data', newfig=False)
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

        # svd data
        nstr = str(self.n)
        title = 'Reconstructed data using ' + nstr + ' components'
        plt.sca(axs[1])
        self.__phelper.plot_heatmap(
            self.svddata, self.time, self.wn,
            title=title, newfig=False)
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

        self.__phelper.plot_traces(self)
        self.__phelper.plot_spectra(self)

    def clean(self):
        """ Unfortunetaly, spyder messes up when the results
            object is invesitgated via the variable explorer.
            Running this method fixes this.

        Returns
        -------
        nothing
        """
        self.__phelper = PlotHelper()


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

    u, s, vt = svd(data)
    return u, s, vt


def show_svs(data, time, wn):
    """ Plots singular values.

    Parameters
    ----------
    data : np.array
        Data matrix subjected to SVD. Assuming *m x n* with m as frequency
        and n as time. But it is actually not important.
    time : np.array
        Time array.
    wn : np.array
        Frequency array.

    Returns
    -------
    nothing
    """

    # ensuring that time spans columns
    if data.shape[1] != time.size:
        data = np.transpose(data)
    u, s, vt = svd(data)
    eig = s**2/np.sum(s**2)

    num = 15
    numlist = list(range(1, num+1))
    varlimits = [0.8, 0.95, 0.995]
    colors = ['red', 'orange', 'forestgreen']
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('First %i singular values' % (num))
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
    plt.show()


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
    res : *mysvd.results()*
        Results object.
    """

    u, s, vt = svd(data)
    if type(n) == int:
        nlist = list(range(n))
    elif type(n) == list:
        n = np.array(n)
        nlist = n-1
    elif type(n) == np.ndarray:
        nlist = n-1

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
    res : *mysvd.results()*
        Results object.
    """

    # ensuring that time spans columns
    if data.shape[1] != time.size:
        data = np.transpose(data)

    # prevents plt.show() from blocking execution
    # plt.ion()
    show_svs(data, time, wn)
    if type(n) == int and n <= 0:
        n = input('How many singular values? ')
        if n.find(',') == -1:
            n = int(n)
        elif n.find(',') > 0:
            [int(i) for i in n.split(',')]
        else:
            print('Wrong input')
            return
    res = reconstruct(data, n)
    res.time = time
    res.wn = wn
    res.plot_results()
    return res
