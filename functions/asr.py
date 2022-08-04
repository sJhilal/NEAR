# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:01:54 2022

@author: 39351
"""
import os

import numpy as np
from numpy.lib.stride_tricks import as_strided

import logging

from scipy import linalg, signal
from statsmodels.robust.scale import mad


from scipy.linalg import toeplitz
from scipy.spatial.distance import cdist, euclidean
from scipy.special import gamma, gammaincinv

try:
    import pyriemann
except ImportError:
    pyriemann = None
    
    
# functions from https://github.com/nbara/python-meegkit 

def mldivide(A, B):
    r"""Matrix left-division (A\B).
    Solves the AX = B for X. In other words, X minimizes norm(A*X - B), the
    length of the vector AX - B:
    - linalg.solve(A, B) if A is square
    - linalg.lstsq(A, B) otherwise
    References
    ----------
    https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html
    """
    try:
        # Note: we must use overwrite_a=False in order to be able to
        # use the fall-back solution below in case a LinAlgError is raised
        return linalg.solve(A, B, assume_a='pos', overwrite_a=False)
    except linalg.LinAlgError:
        # Singular matrix in solving dual problem. Using least-squares
        # solution instead.
        return linalg.lstsq(A, B, lapack_driver='gelsy')[0]
    except linalg.LinAlgError:
        print('Solution not stable. Model not updated!')
        return None

def nonlinear_eigenspace(L, k, alpha=1):
    """Nonlinear eigenvalue problem: total energy minimization.
    This example is motivated in [1]_ and was adapted from the manopt toolbox
    in Matlab.
    TODO : check this
    Parameters
    ----------
    L : array, shape=(n_channels, n_channels)
        Discrete Laplacian operator: the covariance matrix.
    alpha : float
        Given constant for optimization problem.
    k : int
        Determines how many eigenvalues are returned.
    Returns
    -------
    Xsol : array, shape=(n_channels, n_channels)
        Eigenvectors.
    S0 : array
        Eigenvalues.
    References
    ----------
    .. [1] "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems",
       Zhi Zhao, Zheng-Jian Bai, and Xiao-Qing Jin, SIAM Journal on Matrix
       Analysis and Applications, 36(2), 752-774, 2015.
    """
    import pymanopt
    from pymanopt import Problem
    from pymanopt.manifolds import Grassmann
    from pymanopt.optimizers import TrustRegions

    n = L.shape[0]
    assert L.shape[1] == n, 'L must be square.'

    # Grassmann manifold description
    manifold = Grassmann(n, k)
    manifold._dimension = 1  # hack

    # A solver that involves the hessian (check if correct TODO)
    solver = TrustRegions(verbosity=0)

    # Cost function evaluation
    @pymanopt.function.numpy(manifold)
    def cost(X):
        rhoX = np.sum(X ** 2, 1, keepdims=True)  # diag(X*X')
        val = 0.5 * np.trace(X.T @ (L * X)) + \
            (alpha / 4) * (rhoX.T @ mldivide(L, rhoX))
        return val

    # Euclidean gradient evaluation
    @pymanopt.function.numpy(manifold)
    def egrad(X):
        rhoX = np.sum(X ** 2, 1, keepdims=True)  # diag(X*X')
        g = L @ X + alpha * np.diagflat(mldivide(L, rhoX)) @ X
        return g

    # Euclidean Hessian evaluation
    # Note: Manopt automatically converts it to the Riemannian counterpart.
    @pymanopt.function.numpy(manifold)
    def ehess(X, U):
        rhoX = np.sum(X ** 2, 1, keepdims=True)  # np.diag(X * X')
        rhoXdot = 2 * np.sum(X.dot(U), 1)
        h = L @ U + alpha * np.diagflat(mldivide(L, rhoXdot)) @ X + \
            alpha * np.diagflat(mldivide(L, rhoX)) @ U
        return h

    # Initialization as suggested in above referenced paper.
    # randomly generate starting point for svd
    x = np.random.randn(n, k)
    [U, S, V] = linalg.svd(x, full_matrices=False)
    x = U.dot(V.T)
    S0, U0 = linalg.eig(
        L + alpha * np.diagflat(mldivide(L, np.sum(x**2, 1)))
    )

    # Call manoptsolve to automatically call an appropriate solver.
    # Note: it calls the trust regions solver as we have all the required
    # ingredients, namely, gradient and Hessian, information.
    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=egrad,
                      euclidean_hessian=ehess)
    Xsol = solver.run(problem, initial_point=U0)

    return S0, Xsol.point

def fit_eeg_distribution(X, min_clean_fraction=0.25, max_dropout_fraction=0.1,
                         fit_quantiles=[0.022, 0.6],
                         step_sizes=[0.0220, 0.6000],
                         shape_range=np.linspace(1.7, 3.5, 13)):
    """Estimate the mean and SD of clean EEG from contaminated data.
    This function estimates the mean and standard deviation of clean EEG from a
    sample of amplitude values (that have preferably been computed over short
    windows) that may include a large fraction of contaminated samples. The
    clean EEG is assumed to represent a generalized Gaussian component in a
    mixture with near-arbitrary artifact components. By default, at least 25%
    (``min_clean_fraction``) of the data must be clean EEG, and the rest can be
    contaminated. No more than 10% (``max_dropout_fraction``) of the data is
    allowed to come from contaminations that cause lower-than-EEG amplitudes
    (e.g., sensor unplugged). There are no restrictions on artifacts causing
    larger-than-EEG amplitudes, i.e., virtually anything is handled (with the
    exception of a very unlikely type of distribution that combines with the
    clean EEG samples into a larger symmetric generalized Gaussian peak and
    thereby "fools" the estimator). The default parameters should work for a
    wide range of applications but may be adapted to accommodate special
    circumstances.
    The method works by fitting a truncated generalized Gaussian whose
    parameters are constrained by ``min_clean_fraction``,
    ``max_dropout_fraction``, ``fit_quantiles``, and ``shape_range``. The fit
    is performed by a grid search that always finds a close-to-optimal solution
    if the above assumptions are fulfilled.
    Parameters
    ----------
    X : array, shape=(n_channels, n_samples)
        EEG data, possibly containing artifacts.
    max_dropout_fraction : float
        Maximum fraction that can have dropouts. This is the maximum fraction
        of time windows that may have arbitrarily low amplitude (e.g., due to
        the sensors being unplugged) (default=0.25).
    min_clean_fraction : float
        Minimum fraction that needs to be clean. This is the minimum fraction
        of time windows that need to contain essentially uncontaminated EEG
        (default=0.1).
    fit_quantiles : 2-tuple
        Quantile range [lower,upper] of the truncated generalized Gaussian
        distribution that shall be fit to the EEG contents (default=[0.022
        0.6]).
    step_sizes : 2-tuple
        Step size of the grid search; the first value is the stepping of the
        lower bound (which essentially steps over any dropout samples), and the
        second value is the stepping over possible scales (i.e., clean-data
        quantiles) (default=[0.01, 0.01]).
    beta : array
        Range that the clean EEG distribution's shape parameter beta may take.
    Returns
    -------
    mu : array
        Estimated mean of the clean EEG distribution.
    sig : array
        Estimated standard deviation of the clean EEG distribution.
    alpha : float
        Estimated scale parameter of the generalized Gaussian clean EEG
        distribution.
    beta : float
        Estimated shape parameter of the generalized Gaussian clean EEG
        distribution.
    """
    # sort data so we can access quantiles directly
    X = np.sort(X)
    n = len(X)

    # compute z bounds for the truncated standard generalized Gaussian pdf and
    # pdf rescaler
    quants = np.array(fit_quantiles)
    zbounds = []
    rescale = []
    for b in range(len(shape_range)):
        gam = gammaincinv(
            1 / shape_range[b], np.sign(quants - 1 / 2) * (2 * quants - 1))
        zbounds.append(np.sign(quants - 1 / 2) * gam ** (1 / shape_range[b]))
        rescale.append(shape_range[b] / (2 * gamma(1 / shape_range[b])))

    # determine the quantile-dependent limits for the grid search
    # we can generally skip the tail below the lower quantile
    lower_min = np.min(quants)
    # maximum width is the fit interval if all data is clean
    max_width = np.diff(quants)
    # minimum width of the fit interval, as fraction of data
    min_width = min_clean_fraction * max_width

    # Build quantile interval matrix
    cols = np.arange(lower_min,
                     lower_min + max_dropout_fraction + step_sizes[0] * 1e-9,
                     step_sizes[0])
    cols = np.round(n * cols).astype(int)
    rows = np.arange(0, int(np.round(n * max_width)))
    newX = np.zeros((len(rows), len(cols)))
    for i, c in enumerate(range(len(rows))):
        newX[i] = X[c + cols]

    # subtract baseline value for each interval
    X1 = newX[0, :]
    newX = newX - X1
    opt_val = np.inf

    opt_val = np.inf
    opt_lu = np.inf
    opt_bounds = np.inf
    opt_beta = np.inf
    gridsearch = np.round(n * np.arange(max_width, min_width, -step_sizes[1]))
    for m in gridsearch.astype(int):
        mcurr = m - 1
        nbins = int(np.round(3 * np.log2(1 + m / 2)))
        cols = nbins / newX[mcurr]
        H = newX[:m] * cols

        hist_all = []
        for ih in range(len(cols)):
            histcurr = np.histogram(H[:, ih], bins=np.arange(0, nbins + 1))
            hist_all.append(histcurr[0])
        hist_all = np.array(hist_all, dtype=int).T
        hist_all = np.vstack((hist_all, np.zeros(len(cols), dtype=int)))
        logq = np.log(hist_all + 0.01)

        # for each shape value...
        for k, b in enumerate(shape_range):
            bounds = zbounds[k]
            x = bounds[0] + np.arange(0.5, nbins + 0.5) / nbins * np.diff(bounds)  # noqa:E501
            p = np.exp(-np.abs(x) ** b) * rescale[k]
            p = p / np.sum(p)

            # calc KL divergences
            kl = np.sum(p * (np.log(p) - logq[:-1, :].T), axis=1) + np.log(m)

            # update optimal parameters
            min_val = np.min(kl)
            idx = np.argmin(kl)
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = shape_range[k]
                opt_bounds = bounds
                opt_lu = [X1[idx], X1[idx] + newX[m - 1, idx]]

    # recover distribution parameters at optimum
    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)
    mu = opt_lu[0] - opt_bounds[0] * alpha
    beta = opt_beta

    # calculate the distribution's standard deviation from alpha and beta
    sig = np.sqrt((alpha ** 2) * gamma(3 / beta) / gamma(1 / beta))

    return mu, sig, alpha, beta


def yulewalk(order, F, M):
    """Recursive filter design using a least-squares method.
    [B,A] = YULEWALK(N,F,M) finds the N-th order recursive filter
    coefficients B and A such that the filter:
    B(z)   b(1) + b(2)z^-1 + .... + b(n)z^-(n-1)
    ---- = -------------------------------------
    A(z)    1   + a(1)z^-1 + .... + a(n)z^-(n-1)
    matches the magnitude frequency response given by vectors F and M.
    The YULEWALK function performs a least squares fit in the time domain. The
    denominator coefficients {a(1),...,a(NA)} are computed by the so called
    "modified Yule Walker" equations, using NR correlation coefficients
    computed by inverse Fourier transformation of the specified frequency
    response H.
    The numerator is computed by a four step procedure. First, a numerator
    polynomial corresponding to an additive decomposition of the power
    frequency response is computed. Next, the complete frequency response
    corresponding to the numerator and denominator polynomials is evaluated.
    Then a spectral factorization technique is used to obtain the impulse
    response of the filter. Finally, the numerator polynomial is obtained by a
    least squares fit to this impulse response. For a more detailed explanation
    of the algorithm see [1]_.
    Parameters
    ----------
    order : int
        Filter order.
    F : array
        Normalised frequency breakpoints for the filter. The frequencies in F
        must be between 0.0 and 1.0, with 1.0 corresponding to half the sample
        rate. They must be in increasing order and start with 0.0 and end with
        1.0.
    M : array
        Magnitude breakpoints for the filter such that PLOT(F,M) would show a
        plot of the desired frequency response.
    References
    ----------
    .. [1] B. Friedlander and B. Porat, "The Modified Yule-Walker Method of
           ARMA Spectral Estimation," IEEE Transactions on Aerospace Electronic
           Systems, Vol. AES-20, No. 2, pp. 158-173, March 1984.
    Examples
    --------
    Design an 8th-order lowpass filter and overplot the desired
    frequency response with the actual frequency response:
    >>> f = [0, .6, .6, 1]         # Frequency breakpoints
    >>> m = [1, 1, 0, 0]           # Magnitude breakpoints
    >>> [b, a] = yulewalk(8, f, m) # Filter design using a least-squares method
    """
    F = np.asarray(F)
    M = np.asarray(M)
    npt = 512
    lap = np.fix(npt / 25).astype(int)
    mf = F.size
    npt = npt + 1  # For [dc 1 2 ... nyquist].
    Ht = np.array(np.zeros((1, npt)))
    nint = mf - 1
    df = np.diff(F)

    nb = 0
    Ht[0][0] = M[0]
    for i in range(nint):
        if df[i] == 0:
            nb = nb - int(lap / 2)
            ne = nb + lap
        else:
            ne = int(np.fix(F[i + 1] * npt)) - 1

        j = np.arange(nb, ne + 1)
        if ne == nb:
            inc = 0
        else:
            inc = (j - nb) / (ne - nb)

        Ht[0][nb:ne + 1] = np.array(inc * M[i + 1] + (1 - inc) * M[i])
        nb = ne + 1

    Ht = np.concatenate((Ht, Ht[0][-2:0:-1]), axis=None)
    n = Ht.size
    n2 = np.fix((n + 1) / 2)
    nb = order
    nr = 4 * order
    nt = np.arange(0, nr)

    # compute correlation function of magnitude squared response
    R = np.real(np.fft.ifft(Ht * Ht))
    R = R[0:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / (nr - 1)))   # pick NR correlations  # noqa

    # Form window to be used in extracting the right "wing" of two-sided
    # covariance sequence
    Rwindow = np.concatenate(
        (1 / 2, np.ones((1, int(n2 - 1))), np.zeros((1, int(n - n2)))),
        axis=None)
    A = polystab(denf(R, order))  # compute denominator

    # compute additive decomposition
    Qh = numf(np.concatenate((R[0] / 2, R[1:nr]), axis=None), A, order)

    # compute impulse response
    _, Ss = 2 * np.real(signal.freqz(Qh, A, worN=n, whole=True))

    hh = np.fft.ifft(
        np.exp(np.fft.fft(Rwindow * np.fft.ifft(np.log(Ss, dtype=complex))))
    )
    B = np.real(numf(hh[0:nr], A, nb))

    return B, A


def yulewalk_filter(X, sfreq, zi=None, ab=None, axis=-1):
    """Yulewalk filter.
    Parameters
    ----------
    X : array, shape = (n_channels, n_samples)
        Data to filter.
    sfreq : float
        Sampling frequency.
    zi : array, shape=(n_channels, filter_order)
        Initial conditions.
    a, b : 2-tuple | None
        Coefficients of an IIR filter that is used to shape the spectrum of the
        signal when calculating artifact statistics. The output signal does not
        go through this filter. This is an optional way to tune the sensitivity
        of the algorithm to each frequency component of the signal. The default
        filter is less sensitive at alpha and beta frequencies and more
        sensitive at delta (blinks) and gamma (muscle) frequencies.
    axis : int
        Axis to filter on (default=-1, corresponding to samples).
    Returns
    -------
    out : array
        Filtered data.
    zf :  array, shape=(n_channels, filter_order)
        Output filter state.
    """
    [C, S] = X.shape
    if ab is None:
        F = np.array([0, 2, 3, 13, 16, 40, np.minimum(
            80.0, (sfreq / 2.0) - 1.0), sfreq / 2.0]) * 2.0 / sfreq
        M = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
        B, A = yulewalk(8, F, M)
    else:
        A, B = ab

    # apply the signal shaping filter and initialize the IIR filter state
    if zi is None:
        zi = signal.lfilter_zi(B, A)
        zi = np.transpose(X[:, 0] * zi[:, None])
        out, zf = signal.lfilter(B, A, X, zi=zi, axis=axis)
    else:
        out, zf = signal.lfilter(B, A, X, zi=zi, axis=axis)

    return out, zf


def geometric_median(X, tol=1e-5, max_iter=500):
    """Geometric median.
    This code is adapted from [2]_ using the Vardi and Zhang algorithm
    described in [1]_.
    Parameters
    ----------
    X : array, shape=(n_observations, n_variables)
        The data.
    tol : float
        Tolerance (default=1.e-5)
    max_iter : int
        Max number of iterations (default=500):
    Returns
    -------
    y1 : array, shape=(n_variables,)
        Geometric median over X.
    References
    ----------
    .. [1] Vardi, Y., & Zhang, C. H. (2000). The multivariate L1-median and
       associated data depth. Proceedings of the National Academy of Sciences,
       97(4), 1423-1426. https://doi.org/10.1073/pnas.97.4.1423
    .. [2] https://stackoverflow.com/questions/30299267/
    """
    y = np.mean(X, 0)  # initial value

    i = 0
    while i < max_iter:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1. / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < tol:
            return y1

        y = y1
        i += 1
    else:
        print(f"Geometric median could converge in {i} iterations "
              f"with a tolerance of {tol}")

def block_covariance(data, window=128, overlap=0.5, padding=True,
                     estimator='cov'):
    """Compute blockwise covariance.
    Parameters
    ----------
    data : array, shape=(n_chans, n_samples)
        Input data (must be 2D)
    window : int
        Window size.
    overlap : float
        Overlap between successive windows.
    Returns
    -------
    cov : array, shape=(n_blocks, n_chans, n_chans)
        Block covariance.
    """
    from pyriemann.utils.covariance import _check_est

    assert 0 <= overlap < 1, "overlap must be < 1"
    est = _check_est(estimator)
    cov = []
    n_chans, n_samples = data.shape
    if padding:  # pad data with zeros
        pad = np.zeros((n_chans, int(window / 2)))
        data = np.concatenate((pad, data, pad), axis=1)

    jump = int(window * overlap)
    ix = 0
    while (ix + window < n_samples):
        cov.append(est(data[:, ix:ix + window]))
        ix = ix + jump

    return np.array(cov)


def polystab(a):
    """Polynomial stabilization.
    POLYSTAB(A), where A is a vector of polynomial coefficients,
    stabilizes the polynomial with respect to the unit circle;
    roots whose magnitudes are greater than one are reflected
    inside the unit circle.
    Examples
    --------
    Convert a linear-phase filter into a minimum-phase filter with the same
    magnitude response.
    >>> h = fir1(25,0.4);               # Window-based FIR filter design
    >>> flag_linphase = islinphase(h)   # Determines if filter is linear phase
    >>> hmin = polystab(h) * norm(h)/norm(polystab(h));
    >>> flag_minphase = isminphase(hmin)# Determines if filter is minimum phase
    """
    v = np.roots(a)
    i = np.where(v != 0)
    vs = 0.5 * (np.sign(np.abs(v[i]) - 1) + 1)
    v[i] = (1 - vs) * v[i] + vs / np.conj(v[i])
    ind = np.where(a != 0)
    b = a[ind[0][0]] * np.poly(v)

    # Return only real coefficients if input was real:
    if not(np.sum(np.imag(a))):
        b = np.real(b)

    return b


def numf(h, a, nb):
    """Find numerator B given impulse-response h of B/A and denominator A.
    NB is the numerator order.  This function is used by YULEWALK.
    """
    nh = np.max(h.size)
    xn = np.concatenate((1, np.zeros((1, nh - 1))), axis=None)
    impr = signal.lfilter(np.array([1.0]), a, xn)

    b = np.linalg.lstsq(
        toeplitz(impr, np.concatenate((1, np.zeros((1, nb))), axis=None)),
        h.T, rcond=None)[0].T

    return b


def denf(R, na):
    """Compute denominator from covariances.
    A = DENF(R,NA) computes order NA denominator A from covariances
    R(0)...R(nr) using the Modified Yule-Walker method. This function is used
    by YULEWALK.
    """
    
    nr = np.max(np.size(R))
    Rm = toeplitz(R[na:nr - 1], R[na:0:-1])
    Rhs = - R[na + 1:nr]
    A = np.concatenate(
        (1, np.linalg.lstsq(Rm, Rhs.T, rcond=None)[0].T), axis=None)
    return A

class ASR():
    """Artifact Subspace Reconstruction.
    Artifact subspace reconstruction (ASR) is an automatic, online,
    component-based artifact removal method for removing transient or
    large-amplitude artifacts in multi-channel EEG recordings [1]_.
    Parameters
    ----------
    sfreq : float
        Sampling rate of the data, in Hz.
    The following are optional parameters (the key parameter of the method is
    the ``cutoff``):
    cutoff: float
        Standard deviation cutoff for rejection. X portions whose variance
        is larger than this threshold relative to the calibration data are
        considered missing data and will be removed. The most aggressive value
        that can be used without losing too much EEG is 2.5. A quite
        conservative value would be 5 (default=5).
    blocksize : int
        Block size for calculating the robust data covariance and thresholds,
        in samples; allows to reduce the memory and time requirements of the
        robust estimators by this factor (down to Channels x Channels x Samples
        x 16 / Blocksize bytes) (default=10).
    win_len : float
        Window length (s) that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts but
        not shorter than half a cycle of the high-pass filter that was used
        (default=1).
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are going
        to be missed, but is slower (default=0.66).
    max_dropout_fraction : float
        Maximum fraction of windows that can be subject to signal dropouts
        (e.g., sensor unplugged), used for threshold estimation (default=0.1).
    min_clean_fraction : float
        Minimum fraction of windows that need to be clean, used for threshold
        estimation (default=0.25).
    method : {'riemann', 'euclid'}
        Method to use. If riemann, use the riemannian-modified version of
        ASR [2]_.
    memory : float
        Memory size (s), regulates the number of covariance matrices to store.
    estimator : str in {'scm', 'lwf', 'oas', 'mcd'}
        Covariance estimator (default: 'scm' which computes the sample
        covariance). Use 'lwf' if you need regularization (requires pyriemann).
    Attributes
    ----------
    ``state_`` : dict
        Initial state of the ASR filter.
    ``zi_``: array, shape=(n_channels, filter_order)
        Filter initial conditions.
    ``ab_``: 2-tuple
        Coefficients of an IIR filter that is used to shape the spectrum of the
        signal when calculating artifact statistics. The output signal does not
        go through this filter. This is an optional way to tune the sensitivity
        of the algorithm to each frequency component of the signal. The default
        filter is less sensitive at alpha and beta frequencies and more
        sensitive at delta (blinks) and gamma (muscle) frequencies.
    ``cov_`` : array, shape=(channels, channels)
        Previous covariance matrix.
    ``state_`` : dict
        Previous ASR parameters (as derived by :func:`asr_calibrate`) for
        successive calls to :meth:`transform`. Required fields are:
        - ``M`` : Mixing matrix
        - ``T`` : Threshold matrix
        - ``R`` : Reconstruction matrix (array | None)
    References
    ----------
    .. [1] Kothe, C. A. E., & Jung, T. P. (2016). U.S. Patent Application No.
       14/895,440. https://patents.google.com/patent/US20160113587A1/en
    .. [2] Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S.
       (2019). A Riemannian Modification of Artifact Subspace Reconstruction
       for EEG Artifact Handling. Frontiers in Human Neuroscience, 13.
       https://doi.org/10.3389/fnhum.2019.00141
    """

    def __init__(self, sfreq=250, cutoff=5, blocksize=100, win_len=0.5,
                 win_overlap=0.66, max_dropout_fraction=0.1,
                 min_clean_fraction=0.25, name='asrfilter', method='euclid',
                 estimator='scm', **kwargs):

        if pyriemann is None and method == 'riemann':
            logging.warning('Need pyriemann to use riemannian ASR flavor.')
            method = 'euclid'

        self.cutoff = cutoff
        self.blocksize = blocksize
        self.win_len = win_len
        self.win_overlap = win_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.max_bad_chans = 0.3
        self.method = method
        self.memory = int(2 * sfreq)  # smoothing window for covariances
        self.sample_weight = np.geomspace(0.05, 1, num=self.memory + 1)
        self.sfreq = sfreq
        self.estimator = estimator

        self.reset()

    def reset(self):
        """Reset filter."""
        # Initialise yulewalk-filter coefficients with sensible defaults
        F = np.array([0, 2, 3, 13, 16, 40,
                      np.minimum(80.0, (self.sfreq / 2.0) - 1.0),
                      self.sfreq / 2.0]) * 2.0 / self.sfreq
        M = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
        B, A = yulewalk(8, F, M)
        self.ab_ = (A, B)
        self.cov_ = []
        self.zi_ = None
        self.state_ = {}
        self._counter = []
        self._fitted = False

    def fit(self, X, y=None, **kwargs):
        """Calibration for the Artifact Subspace Reconstruction method.
        The input to this data is a multi-channel time series of calibration
        data. In typical uses the calibration data is clean resting EEG data of
        data if the fraction of artifact content is below the breakdown point
        of the robust statistics used for estimation (50% theoretical, ~30%
        practical). If the data has a proportion of more than 30-50% artifacts
        then bad time windows should be removed beforehand. This data is used
        to estimate the thresholds that are used by the ASR processing function
        to identify and remove artifact components.
        The calibration data must have been recorded for the same cap design
        from which data for cleanup will be recorded, and ideally should be
        from the same session and same subject, but it is possible to reuse the
        calibration data from a previous session and montage to the extent that
        the cap is placed in the same location (where loss in accuracy is more
        or less proportional to the mismatch in cap placement).
        Parameters
        ----------
        X : array, shape=(n_channels, n_samples)
            The calibration data should have been high-pass filtered (for
            example at 0.5Hz or 1Hz using a Butterworth IIR filter), and be
            reasonably clean not less than 30 seconds (this method is typically
            used with 1 minute or more).
        """
        if X.ndim == 3:
            X = X.squeeze()

        # Find artifact-free windows first
        clean, sample_mask = clean_windows(
            X,
            sfreq=self.sfreq,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_bad_chans=self.max_bad_chans,
            min_clean_fraction=self.min_clean_fraction,
            max_dropout_fraction=self.max_dropout_fraction)

        # Perform calibration
        M, T = asr_calibrate(
            clean,
            sfreq=self.sfreq,
            cutoff=self.cutoff,
            blocksize=self.blocksize,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            method=self.method,
            estimator=self.estimator)

        self.state_ = dict(M=M, T=T, R=None)
        self._fitted = True

        return clean, sample_mask

    def transform(self, X, y=None, **kwargs):
        """Apply Artifact Subspace Reconstruction.
        Parameters
        ----------
        X : array, shape=([n_trials, ]n_channels, n_samples)
            Raw data.
        Returns
        -------
        out : array, shape=([n_trials, ]n_channels, n_samples)
            Filtered data.
        """
        if X.ndim == 3:
            if X.shape[0] == 1:  # single epoch case
                out = self.transform(X[0])
                return out[None, ...]
            else:
                outs = [self.transform(x) for x in X]
                return np.stack(outs, axis=0)
        else:
            # Yulewalk-filtered data
            X_filt, self.zi_ = yulewalk_filter(
                X, sfreq=self.sfreq, ab=self.ab_, zi=self.zi_)

        if not self._fitted:
            logging.warning('ASR is not fitted ! Returning unfiltered data.')
            return X

        if self.estimator == 'scm':
            cov = 1 / X.shape[-1] * X_filt @ X_filt.T
        else:
            cov = pyriemann.estimation.covariances(X_filt[None, ...],
                                                   self.estimator)[0]

        self._counter.append(X_filt.shape[-1])
        self.cov_.append(cov)

        # Regulate the number of covariance matrices that are stored
        while np.sum(self._counter) > self.memory:
            if len(self.cov_) > 1:
                self.cov_.pop(0)
                self._counter.pop(0)
            else:
                self._counter = [self.memory, ]
                break

        # Exponential covariance weights – the most recent covariance has a
        # weight of 1, while the oldest one in memory has a weight of 5%
        weights = [1, ]
        for c in np.cumsum(self._counter[1:]):
            weights = [self.sample_weight[-c]] + weights

        # Clean data, using covariances weighted by sample_weight
        out, self.state_ = asr_process(X, X_filt, self.state_,
                                       cov=np.stack(self.cov_),
                                       method=self.method,
                                       sample_weight=weights)

        return out


def clean_windows(X, sfreq, max_bad_chans=0.2, zthresholds=[-3.5, 5],
                  win_len=.5, win_overlap=0.66, min_clean_fraction=0.25,
                  max_dropout_fraction=0.1, show=False):
    """Remove periods with abnormally high-power content from continuous data.
    This function cuts segments from the data which contain high-power
    artifacts. Specifically, only windows are retained which have less than a
    certain fraction of "bad" channels, where a channel is bad in a window if
    its power is above or below a given upper/lower threshold (in standard
    deviations from a robust estimate of the EEG power distribution in the
    channel).
    Parameters
    ----------
    X : array, shape=(n_channels, n_samples)
        Continuous data set, assumed to be appropriately high-passed (e.g. >
        1Hz or 0.5Hz - 2.0Hz transition band)
    max_bad_chans : float
        The maximum number or fraction of bad channels that a retained window
        may still contain (more than this and it is removed). Reasonable range
        is 0.05 (very clean output) to 0.3 (very lax cleaning of only coarse
        artifacts) (default=0.2).
    zthresholds : 2-tuple
        The minimum and maximum standard deviations within which the power of
        a channel must lie (relative to a robust estimate of the clean EEG
        power distribution in the channel) for it to be considered "not bad".
        (default=[-3.5, 5]).
    The following are detail parameters that usually do not have to be tuned.
    If you can't get the function to do what you want, you might consider
    adapting these to your data.
    win_len : float
        Window length that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts
        but not shorter than half a cycle of the high-pass filter that was
        used. Default: 1.
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are
        going to be missed, but is slower (default=0.66).
    min_clean_fraction : float
        Minimum fraction that needs to be clean. This is the minimum fraction
        of time windows that need to contain essentially uncontaminated EEG.
        (default=0.25)
    max_dropout_fraction : float
        Maximum fraction that can have dropouts. This is the maximum fraction
        of time windows that may have arbitrarily low amplitude (e.g., due to
        the sensors being unplugged) (default=0.1).
    Returns
    -------
    clean : array, shape=(n_channels, n_samples)
        Dataset with bad time periods removed.
    sample_mask : boolean array, shape=(1, n_samples)
        Mask of retained samples (logical array).
    """
    assert 0 < max_bad_chans < 1, "max_bad_chans must be a fraction !"

    # set internal variables
    truncate_quant = [0.0220, 0.6000]
    step_sizes = [0.01, 0.01]
    shape_range = np.arange(1.7, 3.5, 0.15)
    max_bad_chans = np.round(X.shape[0] * max_bad_chans)

    # set data indices
    [nc, ns] = X.shape
    N = int(win_len * sfreq)
    offsets = np.round(np.arange(0, ns - N, (N * (1 - win_overlap))))
    offsets = offsets.astype(int)
    logging.debug('[ASR] Determining channel-wise rejection thresholds')

    wz = np.zeros((nc, len(offsets)))
    for ichan in range(nc):

        # compute root mean squared amplitude
        x = X[ichan, :] ** 2
        Y = np.array([np.sqrt(np.sum(x[o:o + N]) / N) for o in offsets])

        # fit a distribution to the clean EEG part
        mu, sig, alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction, truncate_quant,
            step_sizes, shape_range)
        # calculate z scores
        wz[ichan] = (Y - mu) / sig

    # sort z scores into quantiles
    wz[np.isnan(wz)] = np.inf  # Nan to inf
    swz = np.sort(wz, axis=0)

    # determine which windows to remove
    if np.max(zthresholds) > 0:
        mask1 = swz[-(int(max_bad_chans) + 1), :] > np.max(zthresholds)
    if np.min(zthresholds) < 0:
        mask2 = (swz[1 + int(max_bad_chans - 1), :] < np.min(zthresholds))

    bad_by_mad = mad(wz, c=1, axis=0) < .1
    bad_by_std = np.std(wz, axis=0) < .1
    mask3 = np.logical_or(bad_by_mad, bad_by_std)

    # combine the three masks
    remove_mask = np.logical_or.reduce((mask1, mask2, mask3))
    removed_wins = np.where(remove_mask)[0]

    # reconstruct the samples to remove
    sample_maskidx = []
    for i, win in enumerate(removed_wins):
        if i == 0:
            sample_maskidx = np.arange(offsets[win], offsets[win] + N)
        else:
            sample_maskidx = np.r_[(sample_maskidx,
                                    np.arange(offsets[win], offsets[win] + N))]

    # delete the bad chunks from the data
    sample_mask2remove = np.unique(sample_maskidx)
    if sample_mask2remove.size:
        clean = np.delete(X, sample_mask2remove, axis=1)
        sample_mask = np.ones((1, ns), dtype=bool)
        sample_mask[0, sample_mask2remove] = False
    else:
        clean = X
        sample_mask = np.ones((1, ns), dtype=bool)

    if show:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(nc, sharex=True, figsize=(8, 5))
        times = np.arange(ns) / float(sfreq)
        for i in range(nc):
            ax[i].fill_between(times, 0, 1, where=sample_mask.flat,
                               transform=ax[i].get_xaxis_transform(),
                               facecolor='none', hatch='...', edgecolor='k',
                               label='selected window')
            ax[i].plot(times, X[i], lw=.5, label='EEG')
            ax[i].set_ylim([-50, 50])
            # ax[i].set_ylabel(raw.ch_names[i])
            ax[i].set_yticks([])
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel(f'ch{i}')
        ax[0].legend(fontsize='small', bbox_to_anchor=(1.04, 1),
                     borderaxespad=0)
        plt.subplots_adjust(hspace=0, right=0.75)
        plt.suptitle('Clean windows')
        plt.show()

    return clean, sample_mask


def asr_calibrate(X, sfreq, cutoff=5, blocksize=100, win_len=0.5,
                  win_overlap=0.66, max_dropout_fraction=0.1,
                  min_clean_fraction=0.25, method='euclid', estimator='scm'):
    """Calibration function for the Artifact Subspace Reconstruction method.
    The input to this data is a multi-channel time series of calibration data.
    In typical uses the calibration data is clean resting EEG data of ca. 1
    minute duration (can also be longer). One can also use on-task data if the
    fraction of artifact content is below the breakdown point of the robust
    statistics used for estimation (50% theoretical, ~30% practical). If the
    data has a proportion of more than 30-50% artifacts then bad time windows
    should be removed beforehand. This data is used to estimate the thresholds
    that are used by the ASR processing function to identify and remove
    artifact components.
    The calibration data must have been recorded for the same cap design from
    which data for cleanup will be recorded, and ideally should be from the
    same session and same subject, but it is possible to reuse the calibration
    data from a previous session and montage to the extent that the cap is
    placed in the same location (where loss in accuracy is more or less
    proportional to the mismatch in cap placement).
    The calibration data should have been high-pass filtered (for example at
    0.5Hz or 1Hz using a Butterworth IIR filter).
    Parameters
    ----------
    X : array, shape=([n_trials, ]n_channels, n_samples)
        *zero-mean* (e.g., high-pass filtered) and reasonably clean EEG of not
        much less than 30 seconds (this method is typically used with 1 minute
        or more).
    sfreq : float
        Sampling rate of the data, in Hz.
    cutoff: float
        Standard deviation cutoff for rejection. X portions whose variance is
        larger than this threshold relative to the calibration data are
        considered missing data and will be removed. The most aggressive value
        that can be used without losing too much EEG is 2.5. A quite
        conservative value would be 5 (default=5).
    blocksize : int
        Block size for calculating the robust data covariance and thresholds,
        in samples; allows to reduce the memory and time requirements of the
        robust estimators by this factor (down to n_chans x n_chans x n_samples
        x 16 / blocksize bytes) (default=100).
    win_len : float
        Window length that is used to check the data for artifact content. This
        is ideally as long as the expected time scale of the artifacts but
        short enough to allow for several 1000 windows to compute statistics
        over (default=0.5).
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are going
        to be missed, but is slower (default=0.66).
    max_dropout_fraction : float
        Maximum fraction of windows that can be subject to signal dropouts
        (e.g., sensor unplugged), used for threshold estimation (default=0.1).
    min_clean_fraction : float
        Minimum fraction of windows that need to be clean, used for threshold
        estimation (default=0.25).
    method : {'euclid', 'riemann'}
        Metric to compute the covariance matrix average.
    Returns
    -------
    M : array
        Mixing matrix.
    T : array
        Threshold matrix.
    """
    logging.debug('[ASR] Calibrating...')

    # set number of channels and number of samples
    [nc, ns] = X.shape

    # filter the data
    X, _zf = yulewalk_filter(X, sfreq, ab=None)

    # window length for calculating thresholds
    N = int(np.round(win_len * sfreq))

    U = block_covariance(X, window=blocksize, overlap=win_overlap,
                         estimator=estimator)
    if method == 'euclid':
        Uavg = geometric_median(U.reshape((-1, nc * nc)))
        Uavg = Uavg.reshape((nc, nc))
    else:  # method == 'riemann'
        Uavg = pyriemann.utils.mean.mean_covariance(U, metric='riemann')

    # get the mixing matrix M
    M = linalg.sqrtm(np.real(Uavg))
    D, Vtmp = linalg.eigh(M)
    # D, Vtmp = nonlinear_eigenspace(M, nc)  TODO
    V = Vtmp[:, np.argsort(D)]

    # get the threshold matrix T
    x = np.abs(np.dot(V.T, X))
    offsets = np.arange(0, ns - N, np.round(N * (1 - win_overlap))).astype(int)

    # go through all the channels and fit the EEG distribution
    mu = np.zeros(nc)
    sig = np.zeros(nc)
    for ichan in reversed(range(nc)):
        rms = x[ichan, :] ** 2
        Y = []
        for o in offsets:
            Y.append(np.sqrt(np.sum(rms[o:o + N]) / N))

        mu[ichan], sig[ichan], alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction)

    T = np.dot(np.diag(mu + cutoff * sig), V.T)
    logging.debug('[ASR] Calibration done.')
    return M, T


def asr_process(X, X_filt, state, cov=None, detrend=False, method='riemann',
                sample_weight=None):
    """Apply Artifact Subspace Reconstruction method.
    This function is used to clean multi-channel signal using the ASR method.
    The required inputs are the data matrix, the sampling rate of the data, and
    the filter state.
    Parameters
    ----------
    X : array, shape=([n_trials, ]n_channels, n_samples)
        Raw data.
    X_filt : array, shape=([n_trials, ]n_channels, n_samples)
        Yulewalk-filtered epochs to estimate covariance. Optional if covariance
        is provided.
    state : dict
        Initial ASR parameters (as derived by :func:`asr_calibrate`):
        - ``M`` : Mixing matrix
        - ``T`` : Threshold matrix
        - ``R`` : Previous reconstruction matrix (array | None)
    cov : array, shape=([n_trials, ]n_channels, n_channels) | None
        Covariance. If None (default), then it is computed from ``X_filt``. If
        a 3D array is provided, the average covariance is computed from all the
        elements in it.
    detrend : bool
        If True, detrend filtered data (default=False).
    method : {'euclid', 'riemann'}
        Metric to compute the covariance matric average.
    Returns
    -------
    clean : array, shape=([n_trials, ]n_channels, n_samples)
        Clean data.
    state : 3-tuple
        Output ASR parameters.
    """
    M, T, R = state.values()
    [nc, ns] = X.shape

    if cov is None:
        if detrend:
            X_filt = signal.detrend(X_filt, axis=1, type='constant')
        cov = block_covariance(X_filt, window=nc ** 2)

    cov = cov.squeeze()
    if cov.ndim == 3:
        if method == 'riemann':
            cov = pyriemann.utils.mean.mean_covariance(
                cov, metric='riemann', sample_weight=sample_weight)
        else:
            cov = geometric_median(cov.reshape((-1, nc * nc)))
            cov = cov.reshape((nc, nc))

    maxdims = int(np.fix(0.66 * nc))  # constant TODO make param

    # do a PCA to find potential artifacts
    if method == 'riemann':
        D, Vtmp = nonlinear_eigenspace(cov, nc)  # TODO
    else:
        D, Vtmp = linalg.eigh(cov)

    V = np.real(Vtmp[:, np.argsort(D)])
    D = np.real(D[np.argsort(D)])

    # determine which components to keep (variance below directional threshold
    # or not admissible for rejection)
    keep = (D < np.sum(np.dot(T, V)**2, axis=0))
    keep += (np.arange(nc) < nc - maxdims)

    # update the reconstruction matrix R (reconstruct artifact components using
    # the mixing matrix)
    if keep.all():
        R = np.eye(nc)  # trivial case
    else:
        VT = np.dot(V.T, M)
        demux = VT * keep[:, None]
        R = np.dot(np.dot(M, linalg.pinv(demux)), V.T)

    if state['R'] is not None:
        # apply the reconstruction to intermediate samples (using raised-cosine
        # blending)
        blend = (1 - np.cos(np.pi * np.arange(ns) / ns)) / 2
        clean = blend * R.dot(X) + (1 - blend) * state['R'].dot(X)
    else:
        clean = R.dot(X)

    state['R'] = R

    return clean, state


def sliding_window(data, window, step=1, axis=-1, copy=True):
    """Calculate a sliding window over a signal.
    Parameters
    ----------
    data : array
        The array to be slided over.
    window : int
        The sliding window size.
    step : int
        The sliding window stepsize (default=1).
    axis : int
        The axis to slide over (defaults=-1).
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data :  array, shape=(..., n_windows, window_size)
        A matrix whose last dimension corresponds to the window size, and the
        second-to-last dimension corresponds to the number of slices.
    Notes
    -----
    Be wary of setting `copy` to `False` as undesired side effects with the
    output values may occur.
    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    """
    if axis >= data.ndim:
        raise ValueError("Axis value out of range")
    if step < 1:
        raise ValueError("Stepsize may not be zero or negative")
    if window > data.shape[axis]:
        print("Sliding window size exceeds size of selected axis")
        return data[..., None]

    shape = list(data.shape)
    shape[axis] = np.floor(
        data.shape[axis] / step - window / step + 1).astype(int)
    shape.append(window)

    strides = list(data.strides)
    strides[axis] *= step
    strides.append(data.strides[axis])
    strided = as_strided(data, shape=shape, strides=strides)

    if copy:
        return strided.copy()
    else:
        return strided