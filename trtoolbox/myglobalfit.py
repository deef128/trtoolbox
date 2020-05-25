# TODO: fix overflow
# TODO: GLA
# TODO: check if back-reactions is nicely implemented
import os
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.optimize import nnls
from scipy.linalg import lstsq
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.mysvd as mysvd
from trtoolbox.plothelper import PlotHelper
# from scipy.special import logsumexp


class Results:
    """ Object containing fit results.

    Attributes
    ----------
    data : np.array
        Data matrix subjected to fitting.
    time : np.array
        Time array.
    wn : np.array
        Wavenumber array.
    offset : str
        Default is *'no'*.
    back : boolean
        Determines if a model with back-reactions was used.
    spectral_offset : np.array
        *optional if offset was chosen*
    method : str
        Chosen method (default is *'svd'*).
    ks : np.array
        Rate constants obtained by fitting.
    var: np.array
        Variance of the rate constants.
    tcs : np.array
        Time constants obtained by fitting.
    das : np.array
        Decay associated spectra.
    profile : np.array
        Concentration profile determined by *ks*
    estimates : np.array
         Contribution of *das* to the dataset for each datapoint.
    fitdata : np.array
        Fitted dataset.
    fittraces : np.array
        *optional if method='svd' was chosen*. Fitted abstract time traces.
    r2 : float
        R^2 of fit.
    """

    def __init__(self):
        self.type = 'gf'
        self.data = np.array([])
        self.time = np.array([])
        self.wn = np.array([])
        self.offset = str()
        self.back = bool
        self.spectral_offset = np.array([])
        self.method = str()
        self.ks = np.array([])
        self.var = np.array([])
        self.tcs = np.array([])
        self.das = np.array([])
        self.profile = np.array([])
        self.estimates = np.array([])
        self.fitdata = np.array([])
        self.fittraces = np.array([])
        self.r2 = 0
        self.wn_name = 'wavenumber'
        self.wn_unit = 'cm^{-1}'
        self.time_name = 'time'
        self.time_unit = 's'
        self._phelper = PlotHelper()

    def init_phelper(self):
        if type(self._phelper) == list:
            self._phelper = PlotHelper()

    def print_results(self):
        """ Prints time constants.
        """

        print('Obtained time constants:')
        if self.back is False:
            for i in range(len(self.tcs)):
                print('%i. %e with variance of %e'
                      % (i+1, self.tcs[i], self.var[i]))
        elif self.back is True:
            for i in range(self.tcs.shape[0]):
                print('%i. forward: %e, backward: %e'
                      % (i+1, self.tcs[i, 0], self.tcs[i, 1]))

    def plot_traces(self):
        """ Plots interactive time traces.

        Returns
        -------
        nothing
        """

        self.init_phelper()
        self._phelper.plot_traces(self)

    def plot_spectra(self, index_alpha=-1, alpha=-1):
        """ Plots interactive spectra.

        Returns
        -------
        nothing
        """

        self.init_phelper()
        self._phelper.plot_spectra(self, index_alpha, alpha)

    def plot_profile(self):
        """ Plots concentration profile.

        Returns
        -------
        nothing
        """

        num_exp = np.shape(self.estimates)[0]
        cm = plt.get_cmap('tab10')
        # nice for choosing aequidistant colors
        # cm = [cm(1.*i/num_exp) for i in range(num_exp)]

        plt.figure()
        for i in range(num_exp):
            plt.plot(
                self.time.reshape(-1),
                self.profile[:, i]*100,
                color=cm(i)
            )
            plt.plot(
                self.time.T,
                self.estimates[i, :]*100,
                'o', markersize=2,
                color=cm(i)
            )
        plt.xscale('log')
        plt.ylabel('concentration / %')
        plt.xlabel('time / s')
        plt.title('Concentration Profile')

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
        # for choosing a specific colormap
        # plt.gca().set_prop_cycle(color=cm)
        plt.plot(self.wn, self.das)
        plt.gca().set_xlim(self.wn[-1], self.wn[0])
        plt.ylabel('absorbance / a.u.')
        plt.xlabel('wavenumber / cm^{-1}')
        plt.title('Decay Associated Spectra')
        plt.legend([str(i+1) for i in range(self.das.shape[0])])

    def plot_fitdata(self):
        """ Plots fitted data.

        Returns
        -------
        nothing
        """

        self.init_phelper()
        title = 'Globally fitted data'
        self._phelper.plot_heatmap(
            self.fitdata, self.time, self.wn,
            title=title, newfig=True)
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_fitdata_3d(self):
        """ 3D plot fitted data.

        Returns
        -------
        nothing
        """

        self.init_phelper()
        title = 'Globally fitted data'
        self._phelper.plot_surface(
            self.fitdata, self.time, self.wn,
            title=title)
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_results(self):
        """ Plots the concentration profile, DAS, fitted data and fitted
            abstract time traces if method='svd' was chosen.

        Returns
        -------
        nothing
        """

        self.plot_profile()
        self.plot_das()
        self.plot_fitdata()

        if self.method == 'svd':
            title = 'Abstract time traces'
            nb_svds = np.shape(self.svdtraces)[0]
            nb_plots = nb_svds
            if self.offset == 'yes':
                nb_plots = nb_svds+1
                title = 'Abstract time traces + offset'
            nb_cols = int(np.ceil(nb_plots/2))
            fig, axs = plt.subplots(2, nb_cols)
            fig.suptitle(title)
            r = 0
            offset = 0
            for i in range(nb_plots):
                if nb_cols == 1:
                    if i < nb_svds:
                        axs[i].plot(self.time.T, self.svdtraces[i, :])
                        axs[i].plot(self.time.T, self.fittraces[i, :])
                        axs[i].set_xscale('log')
                    else:
                        axs[i].plot(self.wn, self.spectral_offset)
                else:
                    if i == nb_cols:
                        r = 1
                        offset = nb_cols
                    if i < nb_svds:
                        axs[r, i-offset].plot(
                            self.time.T, self.svdtraces[i, :])
                        axs[r, i-offset].plot(
                            self.time.T, self.fittraces[i, :])
                        axs[r, i-offset].set_xscale('log')
                    else:
                        axs[r, i-offset].plot(self.wn, self.spectral_offset)

    def clean(self):
        """ Unfortunetaly, spyder messes up when the results
            object is invesitgated via the variable explorer.
            Running this method fixes this.

        Returns
        -------
        nothing
        """
        # self._phelper = PlotHelper()
        # delattr(self, '_phelper')
        self._phelper = []

    def save_to_files(self, path):
        """ Saving results to *.dat files.

        Parameters
        ----------
        path : str
            Path for saving.

        Returns
        -------
        nothing
        """

        if os.path.exists(path) is False:
            answer = input('Path not found. Create (y/n)? ')
            if answer == 'y':
                os.mkdir(path)
            else:
                return

        to_save = ['das', 'data', 'estimates', 'fitdata', 'profile']
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
        tcs_str = ['\n\t%.2e' % (i) for i in self.tcs]
        f.write('Created with trtoolbox\n' +
                '----------------------\n\n' +
                'Obtained time constants: %s\n' % (''.join(tcs_str)) +
                'R^2: %.2f%%\n' % (self.r2) +
                '----------------------\n\n' +
                'Files:\n' +
                '\t- das.dat (Decay associated spectra)\n' +
                '\t- data.dat (Raw data)\n' +
                '\t- estimates.dat (Estimted DAS contributions)\n' +
                '\t- fitdata.dat (Fitted data)\n' +
                '\t- profile.dat (Obtained concentration profile)\n'
                )
        f.close()


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


def model(s, time, ks, back=False):
    """ Creates an array of differential equations according
        to an unidirectional sequential exponential model.
    S[0]/dt = -k0*S[0]
    S[1]/dt = k0*S[0] - k1*S[1]
    S[2]/dt = k1*S[1] - k2*S[2]
    and so on

    Parameters
    ----------
    S : np.array
        Starting concentrations for each species.
    time : np.array
        Time array.
    ks : np.array
        Decay rate constants for each species.
    back : boolean
        Determines if a model with back-reactions is used. If yes,
        ks has to be a matrix with 1st column forward and
        2nd column backward rate constants.
        A -> B with ks[i, 0]
        A <- B with ks[i, 1]
        Therefore ks[-1] is not used.

    Returns
    -------
    arr : np.array
        Array containing the differential equations.
    """

    if back is False:
        arr = [-ks[0] * s[0]]
        for i in range(1, len(ks)):
            arr.append(ks[i-1] * s[i-1] - ks[i] * s[i])

    elif back is True:
        if ks.shape[0] < ks.shape[1]:
            ks = ks.T
        arr = [-ks[0, 0] * s[0] + ks[0, 1] * s[1]]
        for i in range(1, len(ks)-1):
            arr.append(
                ks[i-1, 0] * s[i-1] - ks[i, 0] * s[i]
                - ks[i-1, 1] * s[i] + ks[i, 1] * s[i+1]
                )
        arr.append(ks[-2, 0] * s[-2] - ks[-1, 0] * s[-1] - ks[-2, 1] * s[-1])

    return arr


def create_profile(time, ks, back=False):
    """ Computes a concentration profile according to the *model()* function.

    Parameters
    ----------
    time : np.array
        Time array.
    ks : np.array
        Decay rate constants for each species.

    Returns
    -------
    profile : np.array
        Concentration profile matrix.
    """

    # assuming a starting population of 100% for the first species
    s0 = np.zeros(len(ks))
    s0[0] = 1

    time = time.reshape(-1)
    profile = odeint(model, s0, time, (ks, back))

    return profile


def create_tr(par, time):
    """ Function returning exponential time traces for a given set of parameters.

    Parameters
    ----------
    par : np.array
        Parameter matrix. Last column contains rate constants. Remaining
        columns are prefactors (rate constants x SVD components)
    time : np.array
        Time array.

    Returns
    -------
    profile : np.array
        Concentration profile matrix.
    """

    # sometimes the sum over exponentials encounters an overflow.
    # logsumexp encounters an invalid value error
    old_settings = np.seterr(all='ignore')

    svds = np.shape(par)[1]-1
    time = time.reshape((1, time.size))
    fit_tr = np.empty((svds, time.size))
    for isvds in range(0, svds):
        individual = par[:, isvds] * np.exp(-1*par[:, svds]*time.T)
        fit_tr[isvds, :] = np.sum(individual, axis=1)
    np.seterr(**old_settings)
    return fit_tr


def create_das(profile, data):
    """ Obtains decay associated spectra.

    Parameters
    ----------
    profile : np.array
        Concentration profile matrix
    data : np.array
        Data matrix.

    Returns
    -------
    das : np.array
        Decay associated spectra
    """

    das = lstsq(profile, data.T)
    return das[0].T


def calculate_fitdata(ks, time, data, back=False):
    """ Computes the final fitted dataset.

    Parameters
    ----------
    ks : np.array
        Rate constants
    time : np.array
        Time array.
    data : np.array
        Data matrix.

    Returns
    -------
    fitdata : np.array
        Fitted dataset.
    """

    profile = create_profile(time, ks, back)
    das = create_das(profile, data)
    fitdata = das.dot(profile.T)
    return fitdata


def calculate_estimate(das, data):
    """ Computes contributions of DAS in the raw data.

    Parameters
    ----------
    das : np.array
        DAS
    data : np.array
        Data matrix.

    Returns
    -------
    est : np.array
        Contributions of the individual DAS.
    """

    est = np.empty([np.shape(data)[1], np.shape(das)[1]])
    for i in range(np.shape(data)[1]):
        est[i, :] = nnls(das, data[:, i])[0]
    return est.T


def opt_func_raw(ks, time, data, back):
    """ Optimization function for residuals of fitted data - input data.

    Parameters
    ----------
    ks : np.array
        Rate constants
    time : np.array
        Time array.
    data : np.array
        Data matrix.

    Returns
    -------
    R : np.array
        Flattened array of residuals.
    """

    # deflattens array
    if back is True and ks.ndim == 1:
        ks = ks.reshape(int(ks.shape[0]/2), 2)

    fitdata = calculate_fitdata(ks, time, data, back)
    r = fitdata - data
    return r.flatten()**2


def opt_func_est(ks, time, data, back):
    """ Optimization function for residuals of concentration profile
        and estimated contributions of DAS
    Parameters
    ----------
    ks : np.array
        Rate constants
    time : np.array
        Time array.
    data : np.array
        Data matrix.

    Returns
    -------
    R : np.array
        Flattened array of residuals.
    """

    # deflattens array
    if back is True and ks.ndim == 1:
        ks = ks.reshape(int(ks.shape[0]/2), 2)

    profile = create_profile(time, ks, back)
    das = create_das(profile, data)
    est = calculate_estimate(das, data)
    r = profile.T - est
    return r.flatten()**2


# TODO: check results for back-reactions
def opt_func_svd(par, time, data, svdtraces, nb_exps, back):
    """ Optimization function for residuals of SVD
        abstract time traces - fitted traces.

    Parameters
    ----------
    par : np.array
        Flattened parameter array
    time : np.array
        Time array.
    data : np.array
        Data matrix.
    svdtraces : np.array
        SVD traces
    nb_exps : int
        Number of exponential decay processes. Needed for reshaping the
        flattened paramater array.

    Returns
    -------
    R : np.array
        Flattened array of residuals.
    """

    svds = np.shape(svdtraces)[0]
    par = par.reshape(nb_exps, svds+1)
    r = svdtraces - create_tr(par, time)
    return r.flatten()


def calculate_sigma(res):
    """ Returns the variance of the optimized parameters.

    Parameters
    ----------
    res : *scipy.optimize.OptimizeResult*
        Results object obtained with least squares.

    Returns
    -------
    var : np.array
        Variance of the parameters.
    """

    j = res.jac
    cov = np.linalg.inv(j.T.dot(j))
    var = np.sqrt(np.diagonal(cov))
    return var


def calc_r2(data, res):
    """ Returns R^2 in percent.

    Parameters
    ----------
    data : np.array
        Data matrix.
    res : *scipy.optimize.OptimizeResult*
        Results object obtained with least squares.

    Returns
    -------
    r2 : float
        R^2.
    """

    mean = np.mean(data)
    ss_tot = np.sum((data - mean)**2)
    ss_res = np.sum(res.fun**2)
    r2 = 1 - ss_res/ss_tot
    return r2*100


def doglobalfit(
        data,
        time,
        wn,
        tcs,
        method='svd',
        svds=5,
        offset=False,
        offindex=-1,
        back=False):
    """ Wrapper for global fit routine.

    Parameters
    ----------
    time : np.array
        Time array.
    wn : np.array
        Frequency array.
    data : np.array
        Raw data matrix.
    tcs : np.array
        Initial time constants
    method : np.array
        Method for global fitting.
        *raw* for fitting the residuals of fitted data and input data,
        *est* for fitting the residuals between concentration profile
        and contributions of DAS,
        *svd* for fitting the SVD time traces (default).
    svd : int
        Number of SVD components to be fitted. Default: 5.
    offset : boolean
        Considering the last spectrum to be an offset. Default: False.
    offindex : int
        Index of spectral offset.
    back : boolean
        Determines if a model with back-reactions is used. If yes,
        ks has to be a matrix with 1st column forward and
        2nd column backward rate constants.
        A -> B with ks[i, 0]
        A <- B with ks[i, 1]
        Therefore ks[-1] is not used.

    Returns
    -------
    gf_res : *myglobalfit.results()*
        Results objects.
    """

    data, time, wn = check_input(data, time, wn)
    tcs = np.array(tcs)

    if len(tcs) < 1:
        print('I need at least two time constants.')
        return
    else:
        start_ks = 1./tcs
        # ensuring that start_ks has two columns if back is True
        if back is True and start_ks.shape[1] != 2:
            start_ks = start_ks.T

    if offset is True:
        spectral_offset = data[:, offindex]
        spectral_offset_matrix = np.tile(
            spectral_offset,
            (np.shape(data)[1], 1)
        ).T
        data = data-spectral_offset_matrix

    if method == 'raw':
        res = least_squares(
            opt_func_raw,
            start_ks.flatten(),
            args=(time, data, back)
            )
        if back is False:
            ks = res.x
            var = calculate_sigma(res)
        elif back is True:
            ks = res.x.reshape(start_ks.shape)
            # TODO: variance singular!
            var = -1

    elif method == 'est':
        res = least_squares(
            opt_func_est,
            start_ks.flatten(),
            args=(time, data, back)
            )
        if back is False:
            ks = res.x
            var = calculate_sigma(res)
        elif back is True:
            ks = res.x.reshape(start_ks.shape)
            # TODO: variance singular!
            var = -1

    elif method == 'svd':
        u, s, vt = mysvd.wrapper_svd(data)
        sigma = np.zeros((u.shape[0], vt.shape[0]))
        sigma[:s.shape[0], :s.shape[0]] = np.diag(s)
        svdtraces = sigma[0:svds, :].dot(vt)

        if back is False:
            nb_exps = np.shape(start_ks)[0]
            pars = np.empty((nb_exps, svds+1))
            pars[:, 0:svds] = np.ones((svds,))*0.02
            pars[:, svds] = start_ks.T
        elif back is True:
            nb_exps = start_ks.size
            pars = np.empty((nb_exps, svds+1))
            pars[:, 0:svds] = np.ones((svds,))*0.02
            pars[:, svds] = start_ks.flatten()

        res = least_squares(
            opt_func_svd,
            pars.flatten(),
            args=(time, data, svdtraces, nb_exps, back)
        )
        ks = res.x[svds::svds+1]
        if back is True:
            ks = ks.reshape(start_ks.shape)
        var = calculate_sigma(res)
        var = var[svds::svds+1]

    # gathering results
    gf_res = Results()
    gf_res.offset = offset
    gf_res.back = back
    if offset is True:
        gf_res.data = data+spectral_offset_matrix
        gf_res.spectral_offset = spectral_offset
    else:
        gf_res.data = data
    gf_res.time = time
    gf_res.wn = wn
    gf_res.ks = ks
    gf_res.tcs = 1/ks
    gf_res.var = 1/var
    gf_res.fitdata = calculate_fitdata(ks, time, data, back)
    gf_res.method = method
    gf_res.profile = create_profile(time, ks, back)
    gf_res.das = create_das(gf_res.profile, data)
    gf_res.estimates = calculate_estimate(gf_res.das, data)
    gf_res.r2 = calc_r2(data, res)
    if method == 'svd':
        gf_res.svdtraces = svdtraces
        par = res.x.reshape(nb_exps, svds+1)
        gf_res.fittraces = create_tr(par, time)

    gf_res.print_results()
    print('With an R^2 of %.2f%%' % (gf_res.r2))
    return gf_res
