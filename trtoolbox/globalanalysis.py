# TODO: rate branching

import os
import scipy.special as scsp
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.optimize import nnls
from scipy.linalg import lstsq
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.svd as mysvd
import trtoolbox.expfit as myexpfit
import trtoolbox.pclasses as pclasses


class Results(pclasses.Data):
    """ Object containing fit results.

    Attributes
    ----------
    offset : bool
        True if an offset was used.
    spectral_offset : np.array
        *optional if offset was chosen*
    method : str
        Chosen method (default is *'svd'*).
    rate_constants: RateConstants
        Object containing everything on the rate constants
    xas : np.array
        Decay/Evolution/Species associated spectra.
    profile : np.array
        Concentration profile determined by *ks*
    artefact : bool
        If True, the first two species are merged
    estimates : np.array
         Contribution of *das* to the dataset for each datapoint.
    fitdata : np.array
        Fitted dataset.
    fittraces : np.array
        *optional if method='svd' was chosen*. Fitted abstract time traces.
    svdtraces : np.array
        SVD abstract time traces.
    r2 : float
        R^2 of fit.
    """

    def __init__(self):
        super().__init__()
        self.type = 'gf'
        self.offset = str()
        self.spectral_offset = np.array([])
        self.method = str()
        self.rate_constants = None
        self.xas = np.array([])
        self.profile = np.array([])
        self.artefact = False
        self.estimates = np.array([])
        self.fitdata = np.array([])
        self.fittraces = np.array([])
        self.svdtraces = np.array([])
        self.r2 = 0

    @property
    def tcs(self):
        return self.rate_constants.tcs

    def print_results(self):
        """ Prints time constants.
        """

        tcs = self.rate_constants.tcs
        print('Obtained time constants:')
        if self.rate_constants.style in ['dec', 'seq']:
            for i in range(tcs.shape[0]):
                print('%i. %e with a standard error of %e'
                      % (i+1, tcs[i, 0], self.rate_constants.tcs_err[i]))
        elif self.rate_constants.style == 'back':
            for i in range(tcs.shape[0]):
                print('%i. forward: %e, backward: %e'
                      % (i+1, tcs[i, 0], tcs[i, 1]))
        else:
            print(tcs)

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

    def plot_profile(self):
        """ Plots concentration profile.
        """

        num_exp = np.shape(self.estimates)[1]
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
                self.estimates[:, i]*100,
                'o', markersize=2,
                color=cm(i)
            )
        plt.xscale('log')
        plt.ylabel('concentration / %')
        plt.xlabel('time / s')
        plt.title('Concentration Profile')

    def plot_xas(self):
        """ Plots decay associated spectra.
        """

        plt.figure()
        plt.plot(
            [np.min(self.wn), np.max(self.wn)], [0, 0],
            '--', color='k',
            label='_nolegend_'
        )
        # for choosing a specific colormap
        # plt.gca().set_prop_cycle(color=cm)
        plt.plot(self.wn, self.xas)
        plt.gca().set_xlim(self.wn[-1], self.wn[0])
        plt.ylabel('absorbance / a.u.')
        plt.xlabel('wavenumber / cm^{-1}')
        plt.title('Model Associated Spectra')
        plt.legend([str(i+1) for i in range(self.xas.shape[0])])

    def plot_fitdata(self, interpolate=False, step=.5):
        """ Plots fitted data.

        Parameters
        ----------
        interpolate : boolean
            True for interpolation
        step : float
            Step size for frequency interpolation.
        """

        self.init_phelper()
        title = 'Globally fitted data'
        self._phelper.plot_heatmap(
            self.fitdata, self.time, self.wn,
            title=title, newfig=True,
            interpolate=interpolate, step=step
        )
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_fitdata_3d(self, interpolate=False, step=.5):
        """ 3D plot fitted data.

        Parameters
        ----------
        interpolate : boolean
            True for interpolation
        step : float
            Step size for frequency interpolation.
        """

        self.init_phelper()
        title = 'Globally fitted data'
        self._phelper.plot_surface(
            self.fitdata, self.time, self.wn,
            title=title,
            interpolate=interpolate, step=step
        )
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_results(self):
        """ Plots the concentration profile, DAS, fitted data and fitted
            abstract time traces if method='svd' was chosen.
        """

        self.plot_profile()
        self.plot_xas()
        self.plot_fitdata()

        if 'svd' in self.method:
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
        else:
            title = 'SVD of the residual matrix'
            nb_svds = 4
            nb_cols = 2
            fig, axs = plt.subplots(2, nb_cols)
            fig.suptitle(title)
            resi = self.data - self.fitdata
            u, _, vt = mysvd.wrapper_svd(resi)
            r = 0
            offset = 0
            iu = 0
            iv = 0
            for i in range(nb_svds):
                if i == nb_cols:
                    r = 1
                    offset = nb_cols
                if i < nb_svds:
                    if i - offset == 0:
                        axs[r, i - offset].plot(
                            self.wn, u[:, iu])
                        iu = iu + 1
                    else:
                        axs[r, i - offset].plot(
                            self.time.T, vt[iv, :])
                        axs[r, i - offset].set_xscale('log')
                        iv = iv + 1

    def save_to_files(self, path, comment=''):
        """ Saving results to .dat files.

        Parameters
        ----------
        path : str
            Path for saving.
        comment : str
            Personal comment.
        """

        if os.path.exists(path) is False:
            answer = input('Path not found. Create (y/n)? ')
            if answer == 'y':
                os.mkdir(path)
            else:
                return

        to_save = ['xas', 'data', 'estimates', 'fitdata', 'profile']
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
        if self.tcs.shape[1] == 1:
            tcs_str = ['\n\t%.2e' % i for i in self.tcs]
        else:
            tcs_str = '\n' + str(self.tcs)

        if comment != '':
            comment = '----------------------\n\nComment:\n' + comment

        f.write('Created with trtoolbox\n' +
                '----------------------\n\n' +
                'Model used: %s\n' % self.rate_constants.style +
                'Obtained time constants: %s\n' % (''.join(tcs_str)) +
                'R^2: %.2f%%\n' % self.r2 +
                '----------------------\n\n' +
                'Files:\n' +
                '\t- xas.dat (Decay/Evolution/Species associated spectra)\n' +
                '\t- data.dat (Raw data)\n' +
                '\t- estimates.dat (Estimated DAS contributions)\n' +
                '\t- fitdata.dat (Fitted data)\n' +
                '\t- profile.dat (Obtained concentration profile)\n' +
                comment
                )
        f.close()


class RateConstants:
    """ Container for rate contants related stuff

    Attributes
    ----------
    ks : np.array
        Rate constants
    tcs : np.array
        Time constants
    nb_exps : tuple
        shape of the ks-matrix
    kmatrix : np.array
        K-matrix used for generating differential equations.
    style : str
        Which style of k-matrix was used.
        'dec': parallel decaying processes
        'seq': sequential model
        'back': sequential model with back reactions
        'custom': custom k-matrix
    alphas : np.array
        Defines starting population ratio of 'custom' is choosen
    ks_err : np.array
        Standard error of the fit (rate constants).
    tcs_err : np.array
        Standard error of the fit (time constants).
    """

    def __init__(self, ks):
        if ks.ndim == 1:
            ks = ks.reshape(ks.size, 1)
        self.ks = ks
        self.tcs = convert_tcs(ks)
        self.nb_exps = ks.shape
        self.kmatrix = None
        self.style = None
        self.alphas = None
        self.ks_err = np.array([])
        self.tcs_err = np.array([])

    def set_ks(self, ks):
        """ Sets ks and also tcs accordingly.

        Parameters
        ----------
        ks : np.array
            Rate constants
        """

        if ks.ndim == 1 and self.nb_exps is None:
            ks = ks.reshape(ks.size, 1)
        elif ks.ndim == 1 and self.nb_exps is not None:
            ks = ks.reshape(self.nb_exps)
        self.ks = ks
        self.tcs = convert_tcs(ks)

    def create_kmatrix(self, style=None):
        """ Creates K-matrix. Rate constants should be over rows.

        Parameters
        ----------
        style : str
            Determines how the K-matrix is generated
        """

        if style is None and self.style is not None:
            style = self.style
        elif style is None and self.style is None:
            style = 'seq'

        nb_trans = self.nb_exps[0]
        dec = -1 * np.eye(nb_trans)
        if style == 'seq' and self.ks.shape[1] == 1:
            evo = np.eye(nb_trans, k=-1)
            kmatrix = dec + evo
        elif style == 'dec' and self.ks.shape[1] == 1:
            kmatrix = dec
        elif style == 'back' and self.ks.shape[1] == 2:
            evo = np.eye(nb_trans, k=-1)
            kmatrixf = dec + evo
            kmatrixb = dec + np.eye(nb_trans, k=1)
            kmatrixb[0, 0] = 0
            kmatrix = np.zeros((nb_trans, nb_trans, 2))
            kmatrix[:, :, 0] = kmatrixf
            kmatrix[:, :, 1] = kmatrixb
        else:
            raise Warning(
                'Dimension of the time constant matrix does not match the style. No K-matrix generated'
            )

        self.kmatrix = kmatrix
        self.style = style


def is_square(mat):
    """ Checks if a matrix is square.

    Parameters
    ----------
    mat : np.array
        Matrix

    Returns
    -------
    bool
    """

    return all([len(i) == len(mat) for i in mat])


def convert_tcs(arr):
    """ Converts time to rate constants and vice versa. Necessary due to zeros.

    Parameters
    ----------
    arr : np.array

    Returns
    -------
    np.array
    """

    return np.divide(1, arr, out=np.zeros_like(arr), where=arr != 0)


def model(s, time, rate_constants):
    """ Creates an array of differential equations according to
        (kmatrix * ks).dot(s)
        with ks as rate constants and s as species concentration.

    Parameters
    ----------
    s : np.array
        Starting concentrations for each species.
    time : np.array
        Time array.
    rate_constants : RateConstants
        RateConstants object.

    Returns
    -------
    arr : np.array
        Array containing the differential equations.
    """

    kmatrix = rate_constants.kmatrix
    nb_exps = rate_constants.nb_exps
    style = rate_constants.style
    ks = rate_constants.ks

    if kmatrix is None:
        raise ValueError('No K-matrix')

    if kmatrix.ndim == 2:
        diffs = (kmatrix * ks[:, 0]).dot(s)
    elif kmatrix.ndim == 3 and style == 'back':
        diffs = (kmatrix[:, :, 0] * ks[:, 0]).dot(s)
        diffs = diffs + (kmatrix[:, :, 1] * ks[:, 1]).dot(s)
    elif kmatrix.ndim == 3 and style == 'custom':
        s = s.reshape(nb_exps, order='F')
        diffs = (kmatrix[:, :, 0] * ks[:, 0]).dot(s[:, 0])
        for i in range(1, kmatrix.shape[2]):
            diff2 = (kmatrix[:, :, i] * ks[:, i]).dot(s[:, i])
            diffs = np.vstack((diffs, diff2))
        diffs = diffs.reshape((diffs.size, ))
    else:
        raise ValueError('No suitable K-matrix')

    return diffs


def create_profile(time, rate_constants):
    """ Computes a concentration profile according to the *model()* function.

    Parameters
    ----------
    time : np.array
        Time array.
    rate_constants : RateConstants
        RateConstants object.

    Returns
    -------
    profile : np.array
        Concentration profile matrix.
    """

    ks = rate_constants.ks
    alphas = rate_constants.alphas

    if rate_constants.style == 'dec':
        s0 = np.ones(ks.shape[0])
    elif rate_constants.style == 'custom':
        s0 = np.zeros(ks.shape)
        for i in range(s0.shape[1]):
            s0[0, i] = alphas[i]
        s0 = s0.flatten('F')
    else:
        # assuming a starting population of 100% for the first species
        s0 = np.zeros(ks.shape[0])
        s0[0] = 1

    time = time.reshape(-1)

    # sometimes odeint encounters an overflow
    errs = scsp.geterr()
    errs['overflow'] = 'ignore'
    scsp.seterr(**errs)

    profile = odeint(model, s0, time, (rate_constants, ))

    errs['overflow'] = 'warn'
    scsp.seterr(**errs)

    if rate_constants.style == 'custom':
        profile = np.split(profile, ks.shape[1], axis=1)
        profile = np.sum(profile, axis=0)

    return profile


def create_tr_odeint(rate_constants, pre, time):
    """ Function returning exponential time traces for a given set of parameters.
        Uses odeint function.

    Parameters
    ----------
    rate_constants : RateConstants
        RateConstants object.
    pre : np.array
        Exponential pre-factors.
    time : np.array
        Time array.

    Returns
    -------
    profile : np.array
        Concentration profile matrix.
    """

    if hasattr(rate_constants, 'duplicates'):
        for c in range(rate_constants.duplicates.shape[1]):
            for r in range(rate_constants.duplicates.shape[0]):
                if rate_constants.duplicates[r, c]:
                    rate_constants.ks[r, c+1] = rate_constants.ks[r, 0]

    profile = create_profile(time, rate_constants)
    fit_tr = profile.dot(pre)

    return fit_tr


def create_tr_expfit(rate_constants, pre, time):
    """ Function returning exponential time traces for a given set of parameters.
        Uses expfit module.

        Parameters
        ----------
        rate_constants : RateConstants
            RateConstants object.
        pre : np.array
            Exponential pre-factors.
        time : np.array
            Time array.

        Returns
        -------
        profile : np.array
            Concentration profile matrix.
        """

    fit_tr = np.zeros((time.size, pre.shape[1]))
    for i in range(pre.shape[1]):
        fit_tr[:, i] = myexpfit.create_tr(pre[:, i], 1 / rate_constants.ks, time)
    return fit_tr


def create_xas(profile, data):
    """ Obtains decay associated spectra.

    Parameters
    ----------
    profile : np.array
        Concentration profile matrix
    data : np.array
        Data matrix.

    Returns
    -------
    xas : np.array
        Decay/Evolution/Species associated spectra
    """

    das = lstsq(profile, data.T)
    return das[0].T


def calculate_fitdata(rate_constants, time, data):
    """ Computes the final fitted dataset.

    Parameters
    ----------
    rate_constants : RateConstants
        RateConstants object.
    time : np.array
        Time array.
    data : np.array
        Data matrix.

    Returns
    -------
    fitdata : np.array
        Fitted dataset.
    """

    profile = create_profile(time, rate_constants)
    xas = create_xas(profile, data)
    fitdata = xas.dot(profile.T)
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
    return est


def opt_func_raw(ks, rate_constants, time, data):
    """ Optimization function for residuals of fitted data - input data.

    Parameters
    ----------
    ks : np.array
        Rate constants
    rate_constants : RateConstants
        RateConstants object.
    time : np.array
        Time array.
    data : np.array
        Data matrix.

    Returns
    -------
    R : np.array
        Flattened array of residuals.
    """

    rate_constants.set_ks(ks)

    fitdata = calculate_fitdata(rate_constants, time, data)
    r = fitdata - data
    return r.flatten()


def opt_func_est(ks, rate_constants, time, data):
    """ Optimization function for residuals of concentration profile
        and estimated contributions of DAS

    Parameters
    ----------
    ks : np.array
        Rate constants
    rate_constants : RateConstants
        RateConstants object.
    time : np.array
        Time array.
    data : np.array
        Data matrix.

    Returns
    -------
    R : np.array
        Flattened array of residuals.
    """

    rate_constants.set_ks(ks)

    profile = create_profile(time, rate_constants)
    xas = create_xas(profile, data)
    est = calculate_estimate(xas, data)
    r = profile - est
    return r.flatten()


def opt_func_svd(pars, rate_constants, time, svdtraces, method):
    """ Optimization function for residuals of SVD
        abstract time traces - fitted traces.

    Parameters
    ----------
    pars : np.array
        Flattened parameter array
    rate_constants : RateConstants
        RateConstants object.
    time : np.array
        Time array.
    svdtraces : np.array
        SVD traces
    method : basestring
        Chosen method

    Returns
    -------
    R : np.array
        Flattened array of residuals.
    """

    svds = np.shape(svdtraces)[0]
    nb_exps = rate_constants.nb_exps
    pars = pars.reshape(nb_exps[0], nb_exps[1] + svds)
    rate_constants.set_ks(pars[:, :nb_exps[1]])
    pre = pars[:, nb_exps[1]:]

    if method == 'svd_odeint':
        r = svdtraces.T - create_tr_odeint(rate_constants, pre, time)
    else:   # svd_expfit
        r = svdtraces.T - create_tr_expfit(rate_constants, pre, time)
    return r.flatten()


def calculate_error(res, data):
    """ Returns the standard error of the optimized parameters.

    Parameters
    ----------
    res : scipy.optimize.OptimizeResult
        Results object obtained with least squares.
    data : np.array
        Data matrix.

    Returns
    -------
    perr : np.array
        Standard error of the parameters.
    """

    j = res.jac
    cost = 2 * res.cost  # res.cost is half sum of squares!
    s_sq = cost / (data.size - res.x.size)
    # s_sq = np.var(data - fitdata)

    cov = np.linalg.inv(j.T.dot(j))
    cov = cov * s_sq
    perr = np.sqrt(np.diag(cov))

    # Do Moore-Penrose inverse discarding zero singular values.
    # _, s, VT = svd(res.jac, full_matrices=False)
    # threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    # s = s[s > threshold]
    # VT = VT[:s.size]
    # pcov = np.dot(VT.T / s ** 2, VT)
    # pcov = pcov * s_sq
    # perr = np.sqrt(np.diag(pcov))

    return perr


def calc_r2(data, res):
    """ Returns R^2 in percent.

    Parameters
    ----------
    data : np.array
        Data matrix.
    res : scipy.optimize.OptimizeResult
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


def doglobalanalysis(
        data,
        time,
        wn,
        tcs,
        method='svd_expfit',
        svds=5,
        offset=False,
        offindex=-1,
        style='seq',
        kmatrix=None,
        alphas=None,
        artefact=False,
        silent=False
):
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
        *svd_odeint* for fitting the SVD time traces with odeint function.
        *svd_expfit for fitting the SVD time traces with expfit module (default).
    svds : int
        Number of SVD components to be fitted. Default: 5.
    offset : boolean
        Considering the last spectrum to be an offset. Default: False.
    offindex : int
        Index of spectral offset.
    style : str
        Which style of K-matrix to use.
        'dec': parallel decaying processes
        'seq': sequential model
        'back': sequential model with back reactions. tcs-matrix should have forward time constants in the first
        column and backward in the second.
        'custom': custom k-matrix
    kmatrix : np.array
        K-matrix. Providing an 3D K-matrix is interpreted as parallel reaction pathways. This also useful if
        branching occurs due to heterogeneity. Starting populations can be set with the alpha attribute.
        For more info please see the documentation.
    alphas : np.array
        Sets starting population of the first species.
    artefact : bool
        If True, the first two species are merged
    silent : bool
        Supresses print output

    Returns
    -------
    gf_res : globalanalysis.results
        Results objects.
    """

    data, time, wn = pclasses.check_input(data, time, wn)
    tcs = np.array(tcs)

    # if len(tcs) < 1:
    #     raise ValueError('I need at least two time constants.')
    # else:
    #     start_ks = np.array(1./tcs)
    #     # ensuring that start_ks has two columns if back is True
    #     if style == 'back':
    #         if start_ks.ndim == 1:
    #             raise ValueError('Time constant array dimensions mismatch')
    #         if start_ks.shape[1] != 2 and start_ks.shape[0] == 2:
    #             start_ks = start_ks.T
    #         if start_ks.shape[1] != 2:
    #             raise ValueError('Time constant array dimensions mismatch')
    #

    if offset is True:
        spectral_offset = data[:, offindex]
        spectral_offset_matrix = np.tile(
            spectral_offset,
            (np.shape(data)[1], 1)
        ).T
        data = data-spectral_offset_matrix

    start_ks = convert_tcs(tcs)
    rate_constants = RateConstants(start_ks)

    if kmatrix is None:
        rate_constants.create_kmatrix(style)
    else:
        rate_constants.kmatrix = kmatrix
        rate_constants.alphas = alphas
        rate_constants.style = 'custom'
        method = 'svd_odeint'
        rate_constants.duplicates = np.diff(start_ks, axis=1) == 0

    if method == 'raw':
        res = least_squares(
            opt_func_raw,
            start_ks.flatten(),
            args=(rate_constants, time, data)
            )

    elif method == 'est':
        res = least_squares(
            opt_func_est,
            start_ks.flatten(),
            args=(rate_constants, time, data)
            )

    elif 'svd' in method:
        if method == 'svd_odeint':
            pass
        elif style == 'back':
            method = 'svd_odeint'
        else:
            method = 'svd_expfit'  # standard method with expfit

        svdlist = []
        if type(svds) == int:
            svdlist = list(range(svds))
        elif type(svds) == list:
            svds = np.array(svds)
            svdlist = svds - 1
        elif type(svds) == np.ndarray:
            svdlist = svds - 1
        svds = len(svdlist)

        if any(i < 0 for i in svdlist) is True:
            raise ValueError('Please chose just positive singular values')
        if len(set(svdlist)) != len(svdlist):
            raise ValueError('Please choose different singular values.')

        u, s, vt = mysvd.wrapper_svd(data)
        sigma = np.zeros((u.shape[0], vt.shape[0]))
        sigma[:s.shape[0], :s.shape[0]] = np.diag(s)
        svdtraces = sigma[svdlist, :].dot(vt)

        pars = np.empty((rate_constants.nb_exps[0], svds))
        pars[:, 0:svds] = np.ones((svds,)) * np.max(svdtraces)/2
        pars = np.hstack((rate_constants.ks, pars))

        res = least_squares(
            opt_func_svd,
            pars.flatten(),
            args=(rate_constants, time, svdtraces, method)
        )

        nb_exps = rate_constants.nb_exps
        pars = res.x.reshape(nb_exps[0], nb_exps[1] + svds)

    # gathering results
    gf_res = Results()
    gf_res.offset = offset
    if offset is True:
        gf_res.data = data+spectral_offset_matrix
        gf_res.spectral_offset = spectral_offset
    else:
        gf_res.data = data
    gf_res.time = time
    gf_res.wn = wn
    gf_res.rate_constants = rate_constants
    gf_res.fitdata = calculate_fitdata(rate_constants, time, data)
    if rate_constants.style in ['dec', 'seq']:
        perr = calculate_error(res, data)
        if 'svd' in method:
            perr = perr[0::svds + 1]
        gf_res.rate_constants.ks_err = perr
        gf_res.rate_constants.tcs_err = np.abs(1/(rate_constants.ks[:, 0] + perr) - 1/rate_constants.ks[:, 0])
    else:
        gf_res.rate_constants.ks_err = []
        gf_res.rate_constants.tcs_err = []
    gf_res.method = method
    gf_res.profile = create_profile(time, rate_constants)
    gf_res.artefact = artefact
    if artefact is True:
        merged = gf_res.profile[:, 1:]
        merged[:, 0] = merged[:, 0] + gf_res.profile[:, 0]
        gf_res.profile = merged
    gf_res.xas = create_xas(gf_res.profile, data)
    gf_res.estimates = calculate_estimate(gf_res.xas, data)
    gf_res.r2 = calc_r2(data, res)
    if 'svd' in method:
        gf_res.svdtraces = svdtraces
        gf_res.pre = pars[:, nb_exps[1]:]
        if method == 'svd_odeint':
            gf_res.fittraces = create_tr_odeint(rate_constants, gf_res.pre, time).T
        elif method == 'svd_expfit':
            gf_res.fittraces = create_tr_expfit(rate_constants, gf_res.pre, time).T

    if not silent:
        gf_res.print_results()
        print('With a R^2 of %.2f%%' % gf_res.r2)
    return gf_res
