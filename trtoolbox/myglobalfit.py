from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.optimize import nnls
from scipy.linalg import lstsq
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.mysvd as mysvd


class results:
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
    """
    
    
    def print_results(self):
        """ Prints time constants.
        """
        for i in range(len(self.tcs)):
            print('%e with variance of %e' %(self.tcs[i], self.var[i]))
    
    
    def show_results(self):
        """ Plots the concentration profile, DAS, fitted data and fitted abstract time traces if method='svd' was chosen.
        """
        
        plt.figure()
        plt.plot(self.time.reshape(-1), self.profile*100)
        for i in range(np.shape(self.estimates)[1]):
            plt.scatter(self.time, self.estimates[:,i]*100)
        plt.xscale('log')
        plt.ylabel('concentration / %')
        plt.xlabel('time / s')
        plt.title('Concentration Profile')
        
        plt.figure()
        plt.plot(self.wn, self.das)
        plt.gca().set_xlim(self.wn[-1], self.wn[0])
        plt.ylabel('absorbance / a.u.')
        plt.xlabel('wavenumber / cm^{-1}')
        plt.title('Decay Associated Spectra')
        
        plt.figure()
        plt.pcolormesh(self.time, self.wn, self.fitdata, cmap='jet', shading='gouraud')
        plt.xscale('log')
        plt.ylabel('wavenumber / cm^{-1}')
        plt.xlabel('time / s')
        plt.title('Fitted Data')
        
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
                if i == nb_cols:
                    r = 1
                    offset = nb_cols
                if i < nb_svds:
                    axs[r, i-offset].plot(self.time.T, self.svdtraces[i,:])
                    axs[r, i-offset].plot(self.time.T, self.fittraces[i,:])
                    axs[r, i-offset].set_xscale('log')
                else:
                    axs[r, i-offset].plot(self.wn, self.spectral_offset)


def model(S, time, ks):
    """ Creates an array of differential equations according to a unidirectional sequential exponential model.
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

    Returns
    -------
    arr : np.array
        Array containing the differential equations.
    """
    
    arr = [ -ks[0] * S[0] ]
    for i in range(1, len(ks)):
        arr.append( ks[i-1] * S[i-1] - ks[i] * S[i] )
    return arr


def create_profile(time, ks):
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
    S0 = np.zeros(len(ks))
    S0[0] = 1
    
    time = time.reshape(-1)
    profile = odeint(model, S0, time, (ks,))
    
    return profile


def create_tr(par, time):
    """ Function returning exponential time traces for a given set of parameters.

    Parameters
    ----------
    par : np.array
        Parameter matrix. Last column contains rate constants. Remaining columns are prefactors (rate constants x SVD components)
    time : np.array
        Time array.

    Returns
    -------
    profile : np.array
        Concentration profile matrix.
    """
    
    nb_exps = np.shape(par)[0]
    svds = np.shape(par)[1]-1
    fit_tr = np.empty( (svds, np.shape(time)[1]) )
    for isvds in range(0, svds):
        individual = par[:,isvds] * np.exp(-1*par[:,svds]*time.T)
        fit_tr[isvds,:] = np.sum(individual, axis=1)
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


def calculate_fitdata(ks, time, data):
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
    
    profile = create_profile(time, ks)
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
        est[i,:] = nnls(das, data[:,i])[0]
    return est


def opt_func_raw(ks, time, data):
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
    
    fitdata = calculate_fitdata(ks, time, data)
    R = fitdata - data
    return R.flatten()


def opt_func_est(ks, time, data):
    """ Optimization function for residuals of concentration profile and estimated contributions of DAS
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
    
    profile = create_profile(time, ks)
    das = create_das(profile, data)
    est = calculate_estimate(das, data)
    R = profile - est
    return R.flatten()


def opt_func_svd(par, time, data, svdtraces, nb_exps):
    """ Optimization function for residuals of SVD abstract time traces - fitted traces.

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
        Number of exponential decay processes. Needed for reshaping the flattened paramater array

    Returns
    -------
    R : np.array
        Flattened array of residuals.
    """
    
    svds = np.shape(svdtraces)[0]
    par = par.reshape( nb_exps, svds+1 )
    R = svdtraces - create_tr(par, time)
    return R.flatten()


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
    
    J = res.jac
    cov = np.linalg.inv(J.T.dot(J))
    var = np.sqrt(np.diagonal(cov))
    return var


def doglobalfit(time, wn, data, tcs, method='svd', svds=5, offset=False):
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
        *est* for fitting the residuals between concentration profile and contributions of DAS,
        *svd* for fitting the SVD time traces (default).
    svd : int
        Number of SVD components to be fitted. Default: 5
    offset : boolean
        Considering the last spectrum to be an offset. Default: False

    Returns
    -------
    gf_res : *myglobalfit.results()*
        Results objects.
    """
    
    # ensuring that time spans over columns
    if time.shape[0] != 1:
        time = np.transpose(time)
        wn = np.transpose(wn)
        data = np.transpose(data)
    
    if len(tcs) < 1:
        print('I need at least two time constants.')
        return
    else:
        start_ks = []
        try:
            for tc in tcs:
                start_ks.append(1/float(tc))
                #start_ks.append(float(tc))
        except:
            print(tc)
            print('Just put numbers.')
            return
    start_ks = np.array(start_ks)
    
    # TODO: make offset a fit!
    if offset == True:
        spectral_offset = data[:,-1]
        spectral_offset_matrix = np.tile(spectral_offset, (np.shape(data)[1], 1)).T
        data = data-spectral_offset_matrix
    
    if method == 'raw':
        res = least_squares(opt_func_raw, start_ks, args=(time, data))
        ks = res.x
        var = calculate_sigma(res)
    elif method == 'est':
        res = least_squares(opt_func_est, start_ks, args=(time, data))
        ks = res.x
        var = calculate_sigma(res)
    elif method == 'svd':
        U, s, VT = mysvd.dosvd(data)
        sigma = np.zeros((U.shape[0], VT.shape[0]))
        sigma[:U.shape[0], :U.shape[0]] = np.diag(s)
        svdtraces = sigma[0:svds,:].dot(VT)

        nb_exps = np.shape(start_ks)[0]
        pars = np.empty( (nb_exps,svds+1) )
        pars[:,0:svds] = np.ones((svds,))*0.02
        pars[:,svds] = start_ks.T
        res = least_squares(opt_func_svd, pars.flatten(), args=(time, data, svdtraces, nb_exps))
        ks = res.x[svds::svds+1]
        var = calculate_sigma(res)
        var = var[svds::svds+1]
    
    # gathering results
    gf_res = results()
    gf_res.offset = offset
    if offset == 'yes':
        gf_res.data = data+spectral_offset_matrix
        gf_res.spectral_offset = spectral_offset
    else:
        gf_res.data = data
    gf_res.time = time
    gf_res.wn = wn
    gf_res.ks = ks
    gf_res.tcs = 1/ks
    gf_res.var = var
    gf_res.fitdata = calculate_fitdata(ks, time, data)
    gf_res.method = method
    gf_res.profile = create_profile(time, ks)
    gf_res.das = create_das(gf_res.profile, data)
    gf_res.estimates = calculate_estimate(gf_res.das, data)
    if method == 'svd':
        gf_res.svdtraces = svdtraces
        par = res.x.reshape( nb_exps, svds+1 )
        gf_res.fittraces = create_tr(par, time)
    
    return gf_res
