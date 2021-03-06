# TODO: avg_std

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nrand
from scipy import signal
from trtoolbox.globalanalysis import create_profile, convert_tcs, RateConstants
from trtoolbox.plothelper import PlotHelper


class DataGenerator:
    """ Generates test data consisting of Gaussians.

    Attributes
    ----------
    time : np.array
        Time array.
    wn : np.array
        Wavenumber array.
    data : np.array
        Data matrix.
    rate_constants : globalanalysis.RateConstants
        Object containing information about rate constants.
    sas : np.array
        Species associated spectra.
    profile : np.array
        Concentration profile determined by *tcs*
    """

    def __init__(self):
        self.time = np.array([])
        self.wn = np.array([])
        self.data = np.array([])
        self.rate_constants = None
        self.sas = np.array([])
        self.profile = np.array([])

    @property
    def tcs(self):
        return self.rate_constants.tcs

    def gen_time(self, tlimit=[1e-7, 1e-1], number=500):
        """ Generates time array.

        Parameters
        ----------
        tlimit : list
            Limits for 10^x exponents
        number : int
            Number of time samples.
        """

        self.time = np.logspace(
            np.log10(tlimit[0]),
            np.log10(tlimit[1]),
            number
        )
        self.time = self.time.reshape((1, self.time.size))

    def gen_wn(self, wnlimit=[1500, 1700], step_size=-1):
        """ Generates frequency array. Just takes ints.

        Parameters
        ----------
        wnlimit : list
            Limits for frequency arry.
        step_size : int
            Step size of datapoints. Negative values denote a step size of 1.
        """

        if step_size <= 0:
            step_size = max(wnlimit) - min(wnlimit) + 1
        self.wn = np.linspace(min(wnlimit), max(wnlimit), num=step_size)
        self.wn = self.wn.reshape((self.wn.size, 1))

    def gen_sas(
            self,
            num_das=3,
            num_peaks=1,
            avg_width=40,
            avg_std=5,
            diff=False
            ):
        """ Generates decay associated spectra.

        Parameters
        ----------
        num_das : int
            Number of DAS.
        num_peaks : int
            Number of peaks per DAS.
        avg_width : int
            Average spectral width
        avg_std : float
            Std
        diff : bool
            Peaks can be negative if True.
        """

        self.sas = np.zeros((self.wn.shape[0], num_das))

        # if no peak positions were given
        if type(num_peaks) == int:
            for i in range(num_das):
                das = np.zeros(self.wn.shape[0])
                for _ in range(num_peaks):
                    sc = 1.5 * nrand.random() + 0.5
                    width = int(avg_width * sc)
                    gaus = signal.gaussian(
                        width,
                        std=avg_std,
                        sym=False)
                    pos = nrand.randint(0, high=self.wn.shape[0]-width)
                    if diff is True:
                        pre = nrand.choice([-1, 1])
                    else:
                        pre = 1
                    sc = 1.5 * nrand.random() + 0.5
                    das[pos:pos+width] = das[pos:pos+width] + pre*gaus*sc
                self.sas[:, i] = das.T
        elif isinstance(num_peaks, np.ndarray):         # if positions are given
            if num_peaks.shape[1] != num_das:
                raise ValueError("Dimensions mismatch")
            for i in range(num_das):
                das = np.zeros(self.wn.shape[0])
                for p in range(num_peaks.shape[0]):
                    pos = abs(num_peaks[p, i])
                    if pos == 0:
                        continue
                    width = int(avg_width)
                    gaus = signal.gaussian(
                        width,
                        std=avg_std,
                        sym=False)
                    if pos < np.min(self.wn) or pos > np.max(self.wn):
                        raise ValueError("Peak position not covered by wn!")
                    if num_peaks[p, i] < 0:
                        pre = -1
                    else:
                        pre = 1
                    sc = 1
                    index_pos = np.argmin(abs(self.wn - pos))
                    das[index_pos-width//2:index_pos+width//2] \
                    = das[index_pos-width//2:index_pos+width//2] + pre*gaus*sc
                self.sas[:, i] = das.T

    def gen_tcs(self, tcs=[-1, -1, -1], style='seq'):
        """ Generates time constants

        Parameters
        ----------
        tcs : list or int
            List of tcs. -1 is placeholder for random tcs. Number of
            generated tcs can be specified if tcs is an integer.
        style : str
            Determines the conctruction of the model ('dec', 'seq, 'back')
        """

        if isinstance(tcs, int):
            tcs = [-1 for i in range(tcs)]

        if -1 in tcs:
            min_expo = np.log10(np.min(self.time))
            min_expo = np.ceil(min_expo)

            max_expo = np.log10(np.max(self.time))
            max_expo = np.ceil(max_expo)

            expo = -1 * nrand.choice(
                np.arange(-1*max_expo, -1*min_expo, step=1),
                size=(len(tcs),),
                replace=False)
            expo = np.sort(expo)
            pre = -9 * nrand.random(size=(len(tcs),)) + 9
            gen_tcs = np.array([pre[i]*10.**expo[i] for i in range(len(tcs))])
            tcs = np.array(
                    [gen_tcs[i] if x == -1 else x for i, x in enumerate(tcs)]
            )
        else:
            tcs = np.array(tcs)

        if style == 'back' and tcs.ndim == 1:
            sc = -5 * nrand.random(size=(len(tcs),)) + 5
            back_tcs = np.roll(tcs*sc, 1)
            back_tcs[0] = 0  # no back reaction for the first state
            tcs = np.vstack((tcs, back_tcs)).T

        # self.tcs = np.array([[1e-3, 1e-3], [1e3, 1e-3]])
        ks = convert_tcs(tcs)
        self.rate_constants = RateConstants(ks)
        self.rate_constants.style = style

    def gen_data_sas(self, kmatrix=None):
        """ Generates data.
        """

        if kmatrix is None:
            self.rate_constants.create_kmatrix()
        else:
            self.rate_constants.kmatrix = kmatrix
        self.profile = create_profile(self.time, self.rate_constants)
        self.data = self.sas.dot(self.profile.T)

    def gen_data(
            self,
            tlimit=[1e-7, 1e-1],
            number=500,
            wnlimit=[1500, 1700],
            wnstep=-1,
            tcs=[-1, -1, -1],
            kmatrix=None,
            num_peaks=1,
            avg_width=float(30),
            avg_std=float(5),
            diff=False,
            style='seq',
            noise=False,
            noise_scale=0.1
            ):
        """ Wrapper for data generation.

        Parameters
        ----------
        tlimit : list
            Limits for 10^x exponents
        number : int
            Number of datapoints.
        wnlimit : list
            Limits for frequency arry.
        wnstep : int
            Step size. Negative values denote a step size of 1.
        num_peaks : int
            Number of peaks per DAS.
        avg_width : int
            Average spectral width
        avg_std : float
            Std
        diff : bool
            Peaks can be negative if True.
        tcs : list or int
            List of tcs. -1 is placeholder for random tcs. Number of
            generated tcs can be specified if tcs is an integer.
        kmatrix : np.array
            K-matrix
        style : str
            Determines the conctruction of the model ('dec', 'seq, 'back')
        noise : bool
            Addition of noise.
        noise_scale : float
            Scaling factor for noise.
        """

        if isinstance(tcs, int):
            tcs = [-1 for i in range(tcs)]

        num_das = len(tcs)
        self.gen_time(tlimit, number)
        self.gen_wn(wnlimit, wnstep)
        self.gen_sas(num_das, num_peaks, avg_width, avg_std, diff=diff)
        self.gen_tcs(tcs, style)
        self.gen_data_sas(kmatrix)
        if noise is True:
            self.data = self.data + \
                nrand.normal(0, scale=noise_scale, size=self.data.shape)

    def print_tcs(self):
        """ Prints time constants.
        """

        print('Time constants for data generation:')
        tcs = self.rate_constants.tcs
        if self.rate_constants.style != 'back':
            for i in range(tcs.shape[0]):
                print('%i. %e' % (i+1, tcs[i, 0]))
        else:
            for i in range(tcs.shape[0]):
                print('%i. forward: %e, backward: %e'
                      % (i+1, tcs[i, 0], tcs[i, 1]))

    def plot_profile(self):
        """ Plots concentration profile.
        """

        plt.figure()
        plt.plot(self.time.T, self.profile)
        plt.xscale('log')
        plt.ylabel('concentration / %')
        plt.xlabel('time / s')
        plt.title('Concentration Profile')

    def plot_data(self):
        """ Plots data.
        """

        phelper = PlotHelper()
        title = 'Generated data'
        phelper.plot_heatmap(
            self.data, self.time, self.wn,
            title=title, newfig=True)
        plt.ylabel('frequency')
        plt.xlabel('time')

    def plot_sas(self):
        """ Plots decay associated spectra.
        """

        plt.figure()
        plt.plot(
            [np.min(self.wn), np.max(self.wn)], [0, 0],
            '--', color='k',
            label='_nolegend_'
        )
        plt.plot(self.wn, self.sas, 'o-', markersize=4)
        plt.gca().set_xlim(self.wn[-1], self.wn[0])
        plt.ylabel('absorbance / a.u.')
        plt.xlabel('wavenumber / cm^{-1}')
        plt.title('Species Associated Spectra')
        plt.legend([str(i+1) for i in range(self.sas.shape[0])])
