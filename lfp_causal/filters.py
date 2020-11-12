import numpy as np
import mne

def convolfil(epochs, ntp, type='ones'):
    if isinstance(epochs, str):
        epochs = mne.read_epochs(epochs, preload=True)
    if type == 'ones':
        twin = np.ones(ntp)
    elif type == 'hann':
        twin = np.hanning(ntp)
    elif type == 'hamm':
        twin = np.hamming(ntp)
    elif type == 'bman':
        twin = np.blackman(ntp)
    else:
        raise ValueError('Window type unknown, use: '
                         'ones, hann, hamm or bman instead')

    epo_arr = epochs._data
    for i in range(epo_arr.shape[0]):
        epo = epo_arr[i, :, :].squeeze()
        conv = np.convolve(epo, twin, mode='same')
        conv = np.expand_dims(conv, 0)
        epo_arr[i, :, :] = conv
    epochs._data = epo_arr

    return epochs
