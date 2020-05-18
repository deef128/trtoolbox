# TODO: avg_std
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nrand
from scipy import signal
from trtoolbox.myglobalfit import create_profile
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
    back : boolean
        Determines if a model with back-reactions was used.
    tcs : np.array
        Time constants obtained by fitting.
    das : np.array
        Decay associated spectra.
    profile : np.array
        Concentration profile determined by *tcs*
    """

    def __init__(self):
        self.time = np.array([])
        self.wn = np.array([])
        self.data = np.array([])
        self.das = np.array([])
        self.tcs = np.array([])
        self.profile = np.array([])
        self.back = bool()

    # TODO: positve expos
    def gen_time(self, tlimit=[-7, -1]):
        """ Generates time array. Just works for negative exponents.

        Parameters
        ----------
        tlimit : list
            Limits for 10^x exponents

        Returns
        -------
        nothing
        """

        self.time = np.logspace(tlimit[0], tlimit[1], 500)
        self.time = self.time.reshape((1, self.time.size))

    def gen_wn(self, wnlimit=[1500, 1700], number=-1):
        """ Generates frequency array.

        Parameters
        ----------
        wnlimit : list
            Limits for frequency arry.
        number : int
            Number of datapoints. Negative values denote a step size of 1.

        Returns
        -------
        nothing
        """

        if number <= 0:
            number = max(wnlimit) - min(wnlimit) + 1
        self.wn = np.linspace(min(wnlimit), max(wnlimit), num=number)
        self.wn = self.wn.reshape((self.wn.size, 1))

    def gen_das(
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

        Returns
        -------
        nothing
        """

        self.das = np.zeros((self.wn.shape[0], num_das))

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
            self.das[:, i] = das.T

    # TODO: make that just -1 are replaced
    def gen_tcs(self, tcs=[-1, -1, -1], back=False):
        """ Generates time constants

        Parameters
        ----------
        tcs : list
            List of tcs. -1 is placeholder for random tcs.
        back : bool
            Determines if back reactions are used.

        Returns
        -------
        nothing
        """

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
            self.tcs = np.array([pre[i]*10.**expo[i] for i in range(len(tcs))])
            if back is True:
                sc = -9 * nrand.random(size=(len(tcs),)) + 9
                self.tcs = np.vstack((self.tcs, self.tcs*sc)).T

    def gen_data_das(self, tcs=[-1, -1, -1], back=False):
        """ Generates data.

        Parameters
        ----------
        tcs : list
            List of tcs. -1 is placeholder for random tcs.
        back : bool
            Determines if back reactions are used.

        Returns
        -------
        nothing
        """

        self.profile = create_profile(self.time, 1/self.tcs, back=back)
        self.data = self.das.dot(self.profile.T)
        self.back = back

    def gen_data(
            self,
            tlimit=[-7, -1],
            wnlimit=[1500, 1700],
            tcs=[-1, -1, -1],
            num_das=3,
            num_peaks=1,
            avg_width=30,
            avg_std=5,
            diff=False,
            back=False,
            noise=False,
            noise_scale=0.1
            ):
        """ Wrapper for data generation.

        Parameters
        ----------
        tlimit : list
            Limits for 10^x exponents
        wnlimit : list
            Limits for frequency arry.
        number : int
            Number of datapoints. Negative values denote a step size of 1.
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
        tcs : list
            List of tcs. -1 is placeholder for random tcs.
        back : bool
            Determines if back reactions are used.
        noise : bool
            Addition of noise.
        noise_scale : float
            Scaling factor for noise.

        Returns
        -------
        nothing
        """

        self.gen_time(tlimit)
        self.gen_wn(wnlimit)
        self.gen_das(num_das, num_peaks, avg_width, avg_std, diff=diff)
        self.gen_tcs(tcs, back=back)
        self.gen_data_das(back=back)
        if noise is True:
            self.data = self.data + \
                nrand.normal(0, scale=noise_scale, size=self.data.shape)

    def print_tcs(self):
        """ Prints time constants.

        Returns
        -------
        nothing
        """

        if self.back is False:
            for i in range(len(self.tcs)):
                print('%i. %e' % (i+1, self.tcs[i]))
        elif self.back is True:
            for i in range(len(self.tcs)):
                print('%i. forward: %e, backward: %e'
                      % (i+1, self.tcs[i, 0], self.tcs[i, 1]))

    def plot_profile(self):
        """ Plots concentration profile.

        Returns
        -------
        nothing
        """

        plt.figure()
        plt.plot(self.time.T, self.profile)
        plt.xscale('log')
        plt.ylabel('concentration / %')
        plt.xlabel('time / s')
        plt.title('Concentration Profile')

    def plot_data(self):
        """ Plots data.

        Returns
        -------
        nothing
        """

        phelper = PlotHelper()
        title = 'Generated data'
        phelper.plot_heatmap(
            self.data, self.time, self.wn,
            title=title, newfig=True)
        plt.ylabel('frequency')
        plt.xlabel('time')

    def plot_das(self):
        """ Plots decay associated spectra.

        Returns
        -------
        nothing
        """

        plt.figure()
        plt.plot(
            [np.min(self.wn), np.max(self.wn)], [0, 0],
            '--', color='k',
            label='_nolegend_'
        )
        plt.plot(self.wn, self.das, 'o-', markersize=4)
        plt.gca().set_xlim(self.wn[-1], self.wn[0])
        plt.ylabel('absorbance / a.u.')
        plt.xlabel('wavenumber / cm^{-1}')
        plt.title('Decay Associated Spectra')
        plt.legend([str(i+1) for i in range(self.das.shape[0])])
