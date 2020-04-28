from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class Results:
    """ Object containing fit results.

    Attributes
    ----------
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
    """

    def __init__(self):
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

    def __plot_heatmap(self, data, title='data'):
        """ Plots a nice looking heatmap.

        Parameters
        ----------
        data : np.array
            Data matrix subjected to SVD. Assuming *m x n* with m as frequency
            and n as time. But it is actually not important.
        title : np.array
            Title of plot. Default *data*.

        Returns
        -------
        nothing
        """

        plt.figure()
        # ensuring that time spans columns
        if data.shape[1] != self.time.size:
            data = np.transpose(data)

        if self.time.size == 0 or self.wn.size == 0:
            plt.pcolormesh(data, cmap='jet', shading='gouraud')
        else:
            plt.pcolormesh(
                self.time,
                self.wn,
                data,
                cmap='jet',
                shading='gouraud')
        plt.xscale('log')
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))
        plt.title(title)
        # plt.show()

    def plotdata(self):
        """ Plots a nice looking heatmap of the raw data.

        Parameters
        ----------
        data : np.array
            Data matrix subjected to SVD. Assuming *m x n* with m as frequency
            and n as time. But it is actually not important.

        Returns
        -------
        nothing
        """

        self.__plot_heatmap(self.data, title='Original Data')

    # TODO: slider version
    def plotsvddata(self):
        """ Plots a nice looking heatmap of the reconstructed data.

        Parameters
        ----------
        data : np.array
            Data matrix subjected to SVD. Assuming *m x n* with m as frequency
            and n as time. But it is actually not important.

        Returns
        -------
        nothing
        """

        nstr = str(self.n)
        title = 'Reconstructed data using ' + nstr + ' components'
        self.__plot_heatmap(self.svddata, title=title)

        fig = plt.figure()
        fig.suptitle('Time traces\nblue: Raw, red: SVD')
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.2)
        l1, = plt.plot(self.time, self.data[0, :])
        l2, = plt.plot(self.time, self.svddata[0, :])
        plt.xscale('log')
        ax.margins(x=0)

        axcolor = 'lightgoldenrodyellow'
        axfreq = plt.axes([0.175, 0.05, 0.65, 0.03], facecolor=axcolor)
        sfreq = Slider(
            axfreq, 'Freq',
            np.min(self.wn),
            np.max(self.wn),
            valinit=self.wn[0],
            valstep=abs(self.wn[1]-self.wn[0])
            )

        def update(val):
            val = sfreq.val
            ind = np.where(self.wn == val)[0][0]
            # ax.cla()
            # ax.plot(self.time, self.data[ind, :])
            l1.set_ydata(self.data[ind, :])
            l2.set_ydata(self.svddata[ind, :])
            ymin = min(self.data[ind, :])
            ymax = max(self.data[ind, :])
            sc = 1.1
            ax.set_ylim(ymin*sc, ymax*sc)

        sfreq.on_changed(update)
        plt.show()


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
    numlist = list(range(num))
    varlimits = [0.985, 0.99, 0.995]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('First %i singular values' % (num))
    axs[0].plot(numlist, s[:num], 'o-')
    axs[0].set_title('Singular values')
    axs[0].set_ylabel('|s|')
    axs[1].plot(numlist, np.cumsum(eig[:num])*100, 'o-')
    axs[1].set_title('Cummulative variance explained')
    axs[1].set_ylabel('variance explained / %')
    for limit in varlimits:
        axs[1].plot(numlist, np.ones(num)*limit*100, '--')
        svs = np.where(np.cumsum(eig[:num]) >= limit)[0][0]
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

    plt.ion()
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
    res.plotdata()
    res.plotsvddata()
    plt.show()
    return res
