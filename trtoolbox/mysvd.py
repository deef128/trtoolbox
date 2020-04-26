from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt


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

        title = 'Reconstructed data using %i components' % (self.n)
        self.__plot_heatmap(self.svddata, title=title)


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

    if data.shape[1] != time.size:
        data = np.transpose(data)
    u, s, vt = svd(data)
    plt.figure()
    plt.scatter(list(range(50)), s[0:50])
    plt.title('First 50 singular values')

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


def reconstruct(data, n=0):
    """ Reconstructs data with n singular components.

    Parameters
    ----------
    data : np.array
        Data matrix subjected to SVD. Assuming *m x n* with m as frequency
        and n as time. But it is actually not important.

    Returns
    -------
    res : *mysvd.results()*
        Results object.
    """

    u, s, vt = svd(data)
    if n == 0:
        n = input('How many singular values?')
        n = int(n)

    # create m x n singular values matrix
    sigma = np.zeros((u.shape[0], vt.shape[0]))
    sigma[:s.shape[0], :s.shape[0]] = np.diag(s)

    # reconstruct data
    svddata = u[:, 0:n].dot(sigma[0:n, :].dot(vt))

    res = Results()
    res.data = data
    res.u = u
    res.s = s
    res.vt = vt
    res.n = n
    res.svddata = svddata

    return res


def dosvd(data, time, wn):
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

    Returns
    -------
    res : *mysvd.results()*
        Results object.
    """

    plt.ion()
    show_svs(data, time, wn)
    n = input('How many singular values? ')
    n = int(n)
    res = reconstruct(data, n)
    res.time = time
    res.wn = wn
    res.plotdata()
    res.plotsvddata()
    plt.show()
    return res
    