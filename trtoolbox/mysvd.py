from scipy.linalg import svd
import numpy as np


class Results:
    """ Object containing fit results.

    Attributes
    ----------
    data : np.array
        Data matrix subjected to fitting.
    U : np.array
        U matrix. Represents abstract spectra
    s : np.array
        Singular values.
    VT: np.array
        Transposed V matrix. Represents abstract time traces.
    n : int
        Number of singular components used for data reconstruction.
    svddata : np.array
        Reconstructed data.
    """
    # def __init__(self):
    #     self.U = []
    #     self.s = []
    #     self.VT = []
    #     self.n = 0
    #     self.svddata = []
    pass


def dosvd(data):
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
