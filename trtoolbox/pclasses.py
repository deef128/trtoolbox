import numpy as np
import matplotlib.pyplot as plt
from trtoolbox.plothelper import PlotHelper


def check_input(data, time, wn):
    """ Ensures that all np.arrays have float dtype and that
        time spans over columns, frequency over rows.

    Parameters
    ----------
    data : np.array
        Data matrix.
    time : np.array
        Time array.
    wn : np.array
        Frequency array.

    Returns
    ----------
    data : np.array
        Data matrix.
    time : np.array
        TIme array.
    wn : np.array
        Frequency array.
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

    if data.shape[0] != wn.shape[0] or \
       data.shape[1] != time.shape[1]:
        raise ValueError('Dimensions mismatch!')

    return data, time, wn


class Data:
    """ Parent Data class.

    Attributes
    ----------
    data : np.array
        Data matrix subjected to fitting.
    time : np.array
        Time array.
    time_name : str
        Time name (default: time).
    time_unit : str
        Time uni (default: s).
    wn : np.array
        Frequency array.
    wn_name : str
        Name of the frequency unit (default: wavenumber).
    wn_unit : str
        Frequency unit (default cm^{-1}).
    _phelper : mysvd.PlotHelper
        Plot helper class for interactive plots.
    """

    def __init__(self):
        self.type = 'raw'
        self.data = np.array([])
        self.time = np.array([])
        self.time_name = 'time'
        self.time_unit = 's'
        self.wn = np.array([])
        self.wn_name = 'wavenumber'
        self.wn_unit = 'cm^{-1}'
        self._phelper = PlotHelper()

    def check_input(self):
        self.data, self.time, self.wn = check_input(self.data, self.time, self.wn)

    def clean(self):
        """ Unfortunetaly, spyder messes up when the results
            object is invesitgated via the variable explorer.
            Running this method fixes this.
        """
        self._phelper = []

    def init_phelper(self):
        """ Initiliazes phelper after clean().
        """

        if type(self._phelper) == list:
            self._phelper = PlotHelper()

    def plot_rawdata(self, interpolate=False, step=.5):
        """ Plots raw data.

        Parameters
        ----------
        interpolate : boolean
            True for interpolation
        step : float
            Step size for frequency interpolation.
        """

        self.init_phelper()
        title = 'Raw data'
        self._phelper.plot_heatmap(
            self.data, self.time, self.wn,
            title=title, newfig=True,
            interpolate=interpolate, step=step
        )
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_rawdata_3d(self, interpolate=False, step=.5):
        """ 3D plot raw data.

        Parameters
        ----------
        interpolate : boolean
            True for interpolation
        step : float
            Step size for frequency interpolation.
        """

        self.init_phelper()
        title = 'Raw data'
        self._phelper.plot_surface(
            self.data, self.time, self.wn,
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
