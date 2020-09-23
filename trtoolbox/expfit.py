# TODO: check variance

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


class Results:
    """ Object containing fit results

    Attributes
    ----------
    data : np.array
        Time trace subjected to fitting.
    time : np.array
        Time array
    tcs : np.array
        Time constants
    pre : np.array
        Prefactors
    var : np.array
        Variance of tcs
    traces : np.array
        Individual expontential fit traces
    fit : np.array
        Fitted time trace
    """

    def __init__(self, pre, tcs, time):
        self.data = None
        self.time = time
        self.tcs = tcs
        self.pre = pre
        self.var = None
        self.traces = np.zeros((self.tcs.size, self.time.size))
        self.create_traces()
        self.fit = create_tr(self.pre, self.tcs, self.time)

    def print_results(self):
        """ Prints time constants.
        """
        print('Obtained time constants:')
        for i, tc in enumerate(self.tcs):
            print('%i. %e' % (i+1, tc))

    def create_traces(self):
        """ Creates individual exponential traces
        """

        for i, tc in enumerate(self.tcs):
            self.traces[i, :] = self.pre[i] * np.exp(-1/tc * self.time)

    def plot_results_traces(self):
        """ Plots individual exponential traces.
        """

        plt.figure()
        plt.plot(self.time, self.data, 'o-', markersize=0.5)
        for tr in self.traces:
            plt.plot(self.time, tr)
        plt.xscale('log')

    def plot_results(self):
        """ Plots result.
        """

        self.print_results()
        plt.figure()
        plt.plot(self.time, self.data, 'o', markersize=2)
        plt.plot(self.time, self.fit)
        plt.xscale('log')


# TODO: overflow
def create_tr(pre, tcs, time):
    """ Creates fitted time trace

    Parameters
    ----------
    pre : np.array
        Prefactors
    tcs : np.array
        Time constants
    time : np.array
        Time array

    Returns
    -------
    tr : np.array
        Fitted time trace
    """

    tr = np.zeros(time.size)
    for ele in zip(pre, tcs):
        tr = tr + ele[0]*np.exp(-1/ele[1] * time)
    return tr


def opt_func(pre_plus_tcs, data, time):
    """ Optimization function

    Parameters
    ----------
    pre_plus_tcs : np.array
        Prefactors first column, time constants second
    data : np.array
        Time trace subjected to fitting
    time : np.array
        Time array

    Returns
    -------
    r : np.array
        Residuals
    """

    nb_exps = int(pre_plus_tcs.size / 2)
    pre_plus_tcs = pre_plus_tcs.reshape((nb_exps, 2))
    pre = pre_plus_tcs[:, 0]
    tcs = pre_plus_tcs[:, 1]
    r = data - create_tr(pre, tcs, time)
    return r.flatten()


def calculate_sigma(res):
    """ Returns the variance of the optimized parameters.

    Parameters
    ----------
    res : scipy.optimize.OptimizeResult
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


def dofit(data, time, init):
    """ Do exponential fitting

    Parameters
    ----------
    data : np.array
        Time trace subjected to fitting
    time : np.array
        Time array
    init : np.array
        Initial guesses. Prefactors first column, time constants second

    Returns
    -------
    res : self.Results()
        Results object
    """

    fitres = least_squares(opt_func, init.flatten(), args=(data, time))
    x = fitres.x.reshape(init.shape)
    res = Results(x[:, 0], x[:, 1], time)
    var = calculate_sigma(fitres)
    res.var = var.reshape(init.shape)[:, 1]
    res.data = data
    return res
