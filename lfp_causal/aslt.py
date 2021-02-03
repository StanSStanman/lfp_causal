import sys
import numpy as np
import xarray as xr

try:
    import cupy as cp
    from cusignal.convolution.convolve import fftconvolve
except:
    cp = np
    from scipy.signal import fftconvolve

    cp.asnumpy = np.asarray


def bw_cf(t, bw, cf):
    """Computes the complex wavelet coefficients for the desired time point t,
    bandwidth bw and center frequency cf"""
    cnorm = 1 / (bw * np.sqrt(2 * np.pi))
    exp1 = cnorm * np.exp(-(t ** 2) / (2 * (bw ** 2)));
    res = np.exp(2j * np.pi * cf * t) * exp1
    return res


def gauss(t, sd):
    """Compute the gaussian coefficient for the desired time point t and
    standard deviation sd"""
    cnorm = 1 / (sd * np.sqrt(2 * np.pi))
    res = cnorm * np.exp(-(t ** 2) / (2 * (sd ** 2)))
    return res


def cxmorlet(fc, n_cycles, sfreq):
    """Computes the complex Morlet wavelet for the desired center frequency.

    Parameters
    ----------
    fc : center frequency
    n_cycles : number of cycles
    sfreq : sampling frequency
    """
    # we want to have the last peak at 2.5 SD
    sd = (n_cycles / 2) * (1 / fc) / 2.5
    wl = int(2 * np.floor(np.fix(6 * sd * sfreq) / 2) + 1)
    w = np.zeros((wl), dtype=np.complex128)
    gi = 0
    off = np.fix(wl / 2)
    for i in range(wl):
        t = (i - off) / sfreq
        w[i] = bw_cf(t, sd, fc)
        gi += gauss(t, sd)
    w /= gi
    return w


def aslt(data, sfreq, foi, n_cycles, order=None, mult=False):
    """Adaptive superresolution wavelet (superlet) transform.

    - data (array_like) : (n_epochs, n_times)
    - sfreq (float) : sampling frequency
    - foi (array_like) : central frequency of interest
    - n_cycles (integer) : number of initial wavelet cycles
    - order (array_like) : interval of super-resolution orders of shape (2,).
      For example, use order=[1, 30]
    - mult (bool) : specifies the use of multiplicative super-resolution (True)
      or additive (False)
    """
    # inputs checking
    assert isinstance(data, np.ndarray)
    data = np.atleast_2d(data).astype(np.float32)
    n_epochs, n_times = data.shape
    foi = np.asarray(foi)
    n_freqs = len(foi)
    if isinstance(n_cycles, (int, float)):
        n_cycles = np.full_like(foi, n_cycles)

    # check order parameter and initialize the order used at each frequency. If
    # empty, go with an order of 1 for each frequency (single wavelet per set)
    if order is not None:
        order_ls = np.fix(np.linspace(order[0], order[1], n_freqs)).astype(int)
    else:
        order_ls = np.ones((n_freqs,), dtype=np.int)

    # the padding will be size of the lateral zero-pads, which serve to avoid
    # border effects during convolution
    padding = 0

    # the wavelet sets
    wavelets = dict()

    # initialize wavelet sets for either additive or multiplicative
    # superresolution
    for i_freq in range(n_freqs):
        for i_ord in range(order_ls[i_freq]):
            # get the number of cycles
            if mult:  # multiplicative superresolution
                n_cyc = n_cycles[i_freq] * (i_ord + 1)
            else:  # additive superresolution
                n_cyc = n_cycles[i_freq] + i_ord

            # each new wavelet has n_cyc extra cycles
            _w = cxmorlet(foi[i_freq], n_cyc, sfreq)

            # the margin will be the half-size of the largest wavelet
            padding = max(padding, np.fix(len(_w) / 2))

            wavelets[(i_freq, i_ord)] = _w

    # the zero-padded buffer
    buffer = cp.zeros((n_epochs, int(n_times + 2 * padding)),
                      dtype=cp.float32)

    # convenience indexers for the zero-padded buffer
    bufbegin = int(padding)
    bufend = int(padding + n_times)

    # fill the central part of the buffer with input data
    buffer[:, bufbegin:bufend] = cp.asarray(data)

    # the output scalogram
    wtresult = cp.zeros((n_epochs, n_freqs, n_times), dtype=cp.float32)

    for i_freq in range(n_freqs):
        # pooling buffer, starts with 1 because we're doing geometric mean
        temp = cp.ones((n_epochs, n_times), dtype=cp.float32)

        # compute the convolution of the buffer with each wavelet in the
        # current set
        for i_ord in range(order_ls[i_freq]):
            # get the single wavelets
            sw = cp.asarray(wavelets[(i_freq, i_ord)]).reshape(1, -1)

            # restricted convolution (input size == output size)
            _temp = fftconvolve(buffer, sw, mode='same', axes=1)

            # accumulate the magnitude (times 2 to get the full spectral
            # energy
            temp *= (2 * cp.abs(_temp[:, bufbegin:bufend]))

        # compute the power of the geometric mean
        root = 1. / float(order_ls[i_freq])
        temp = (temp ** root) ** 2

        # accumulate the current FOI to the result spectrum
        wtresult[:, i_freq, :] += temp

    if 'cupy' not in sys.modules:
        return cp.asarray(wtresult)
    else:
        return cp.asnumpy(wtresult)


def parallel_aslt(data, sfreq, freqs, n_cycles, times,
                  order=None, mult=False, n_jobs=-1):
    n_trials = data.shape[0]
    from joblib import Parallel, delayed
    fpow = Parallel(n_jobs=n_jobs, verbose=0) \
        (delayed(aslt)(data, sfreq, [_foi], _nc, order, mult)
        for _foi, _nc in zip(freqs, n_cycles))

    fpow = xr.DataArray(np.concatenate(fpow, axis=1),
                        coords=(range(n_trials), freqs, times),
                        dims=['trials', 'freqs', 'times'])

    return fpow

if __name__ == '__main__':
    times = np.arange(-1.3, 1.5, 0.001)
    d = np.random.uniform(np.random.normal(loc=0, scale=1,
                                           size=(20, len(times))))
    sfreq = 1000
    foi = np.linspace(5, 120, 80)
    n_cycles = foi / 2.
    order = [1, 30]
    mult = True

    parallel_aslt(d, sfreq, foi, n_cycles, times, order, mult)
