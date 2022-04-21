import mne
import numpy as np
from scipy.stats import zscore
from brainets.spectral import mt_hga
from lfp_causal.directories import *
from lfp_causal.old.controls import session_name
import time


def time_frequency_analysis(subject=None, condition=None, session=None, event=None, epochs=None, band='hga'):
    assert (subject, epochs) != (None, None), 'You should insert at least one source for epochs.'
    if not isinstance(epochs, (mne.Epochs, mne.BaseEpochs, mne.epochs.Epochs, mne.epochs.BaseEpochs)):
        epochs = mne.read_epochs(os.path.join(epochs_dir.format(subject, condition, session), '{0}_{1}-epo.fif'.format(session, event)))
    f_bands = {'beta': (20., 4, 4), 'gamma': (36., 10, 6), 'hga': (100., 16, 12)}
    freq, n_cy, t_bd = f_bands[band]
    tfr = mt_hga(epochs, f=freq, n_cycles=n_cy, time_bandwidth=t_bd, log_transform=False)
    return tfr


def z_score_tfr(epo_tfr, baseline=False):
    assert isinstance(epo_tfr, mne.time_frequency.EpochsTFR), 'The variable must be an EpocsTFR object.'
    z_tfr = epo_tfr.copy()
    if baseline == False:
        for e in range(z_tfr.data.shape[0]):
            for c in range(z_tfr.data.shape[1]):
                for f in range(z_tfr.data.shape[2]):
                    z_tfr.data[e, c, f, :] = zscore(z_tfr.data[e, c, f, :])
    if isinstance(baseline, tuple):
        assert len(baseline) == 2, 'Baseline needs a tuple of two timepoints (tmin, tmax).'
        bline = epo_tfr.copy()
        bline.crop(baseline[0], baseline[1])
        avg = bline.data.mean(-1, keepdims=True)
        std = bline.data.std(-1, keepdims=True)
        z_tfr.data = (z_tfr.data - avg) / std
    return z_tfr


def plot_psd(subject, condition, session):

    # Correct session name and read the associate raw file
    trial_num = session_name(session)
    raw_fname = raw_dir.format(subject, condition) + '{0}_raw.fif'.format(trial_num)
    raw = mne.io.read_raw_fif(raw_fname, preload=True)

    raw.plot_psd(fmax=200)


def plot_tfr(subject, condition, session, event):

    # Correct session name and read the associate epoch file
    trial_num = session_name(session)
    epochs_fname = epochs_dir.format(subject, condition) + '{0}\\{0}_{1}-epo.fif'.format(trial_num, event)
    epochs = mne.read_epochs(epochs_fname, preload=True)

    # tmin, tmax = np.round(epochs.tmin, decimals=1)+0.2, np.round(epochs.tmax, decimals=1)-0.2
    baseline  = epochs.baseline # (0.3, 0.5)
    # decim = 1000#np.round(epochs.info['sfreq']).astype(int) #* 1000
    freqs = np.arange(50.0, 150.0, 10.0)
    n_cycles = freqs / 5
    start = time.time()
    tfr, itc = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles, return_itc=True, n_jobs=-1, average=True)
    end = time.time()
    tmin, tmax = np.round(tfr.times[0], decimals=1) + 0.2, np.round(tfr.times[-1], decimals=1) - 0.2
    tfr.plot(baseline=baseline, mode='zlogratio', tmin=tmin, tmax=tmax, vmin=-3, vmax=3)
    itc.plot(baseline=baseline, mode='zlogratio', tmin=tmin, tmax=tmax, vmin=-3, vmax=3)
    print(end-start)
    # tfr.plot([0], baseline=(None, None), mode='zlogratio', tmin=tmin, tmax=tmax, vmin=-3, vmax=3)
    # tfr.plot([0], baseline=(None, None), mode='zlogratio', vmin=-3, vmax=3)

# if __name__ == '__main__':
#     plot_psd()