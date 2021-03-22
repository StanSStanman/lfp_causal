import numpy as np
from scipy.interpolate import interp1d


def despike_lfp(data, sf, t_spk, len_spk):
    '''
    Algorithm to despike LFP datas

    Parameters
    ----------
    data: array_like
        LFP data
    sf: float
        sampling frequency
    t_spk: array_like
        time of the spikes in seconds
    len_spk: int | float | array_like
        total length of the spike in seconds

    Returns
    -------
    despk: array_like
        despiked array

    '''
    # spikes to n. of samples (points in vector)
    spk_smpl = np.round(t_spk * sf).astype(int)

    # spike time window to n. of samples
    if isinstance(len_spk, (int, float)):
        len_spk = np.full(len(spk_smpl), len_spk).astype('float32')
    spk_tw = np.round((len_spk * sf) / 2.).astype(int)

    dnan = data.copy()
    for st, stw in zip(spk_smpl, spk_tw):
        rnan = range(st - stw, st + stw)
        if rnan[0] > 0 and rnan[-1] < len(data):
            dnan[rnan] = np.nan

    v = np.arange(len(dnan))
    xq = v.copy()
    v = v[~np.isnan(dnan)]
    dnan = dnan[~np.isnan(dnan)]

    func = interp1d(v, dnan, kind='slinear')
    despk = func(xq)

    return despk


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Fs = 8000
    f = 5
    sample = 8000
    x = np.arange(sample)
    y = np.sin(2 * np.pi * f * x / Fs)

    spk_pos = np.random.randint(0, sample, 10)
    spk_val = np.random.uniform(-0.5, 0.5, 10)

    y[spk_pos] = y[spk_pos] + spk_val
    plt.plot(x, y)

    _y = despike_lfp(y, 1, spk_pos, 2)
    plt.plot(x, _y)

    plt.show()
