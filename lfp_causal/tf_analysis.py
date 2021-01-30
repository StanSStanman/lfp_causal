import mne
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from lfp_causal.epochs import reject_bad_epochs
# import cmasher as cmr


def epochs_tf_analysis(epochs, picks=None, t_win=(-.5, .5), freqs=None,
                       n_cycles=None, avg=True, baseline=None, show=True):
    if isinstance(epochs, str):
        epochs = mne.read_epochs(epochs, preload=True)
    if picks is not None:
        ch_names = picks
    else:
        ch_names = epochs.ch_names

    for ch in ch_names:
        if 'LFP' in ch:
            ch_epo = epochs.copy().pick([ch])

            power, itc = time_freq_analysis(ch_epo, freqs, n_cycles, avg)

            fig = plot_tf(power, itc, t_window=t_win, baseline=baseline,
                          show=show)

    return power, itc, fig


def evoked_tf_analysis(evoked, t_win=(-.5, .5), freqs=None,
                       n_cycles=None, baseline=None, show=True):
    if isinstance(evoked, str):
        evoked = mne.read_evokeds(evoked, preload=True)
    if freqs is None:
        fmin = evoked.info['highpass']
        fmax = evoked.info['lowpass']
    if isinstance(freqs, tuple):
        fmin, fmax = freqs
    if not isinstance(freqs, np.ndarray):
        freqs = freqs_bins(fmin, fmax)

    if n_cycles is None:
        n_cycles = freqs / 2.

    power = mne.time_frequency.tfr_morlet(evoked, freqs=freqs,
                                          n_cycles=n_cycles,
                                          return_itc=False, n_jobs=-1,
                                          average=True)

    fig = plot_tf(power, None, t_window=t_win, baseline=baseline,
                  show=show)

    return power, fig


def freqs_bins(fmin, fmax, mul=2):
    freq_band = {'delta': (1, 6), 'theta': (5, 9), 'alpha': (8, 14),
                 'beta': (13, 31), 'gamma': (30, 51), 'high gamma': (50, 121)}
    fmin, fmax = int(fmin), int(fmax)
    freqs = []
    for k in freq_band.keys():
        b = freq_band[k]
        if fmin < b[0] and fmax > b[1]:
            freqs.append(np.logspace(*np.log10([b[0], b[1]]),
                                     num=len(range(*b)) * mul))
        elif fmin in range(*b) and fmax in range(*b):
            freqs.append(np.logspace(*np.log10([fmin, fmax]),
                                     num=len(range(fmin, fmax)) * mul))
        elif fmin in range(*b) and fmax not in range(*b):
            freqs.append(np.logspace(*np.log10([fmin, b[1]]),
                                     num=len(range(fmin, b[1])) * mul))
        elif fmin not in range(*b) and fmax in range(*b):
            freqs.append(np.logspace(*np.log10([b[0], fmax]),
                                     num=len(range(b[0], fmax)) * mul))
    freqs = np.unique(np.hstack(tuple(freqs)))

    return freqs


def time_freq_analysis(epochs, freqs=None, n_cycles=None, avg=True):
    if freqs is None:
        fmin = epochs.info['highpass']
        fmax = epochs.info['lowpass']
        # freqs = (fmin, fmax)
    if isinstance(freqs, tuple):
        fmin, fmax = freqs
    if not isinstance(freqs, np.ndarray):
        freqs = freqs_bins(fmin, fmax)

    if n_cycles is None:
        n_cycles = freqs / 2.

    if avg:
        power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs,
                                                   n_cycles=n_cycles,
                                                   return_itc=True, n_jobs=-1,
                                                   average=True,
                                                   verbose='DEBUG')
        return power, itc

    elif not avg:
        power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs,
                                              n_cycles=n_cycles,
                                              return_itc=False, n_jobs=-1,
                                              average=False,
                                              verbose='DEBUG')
        return power, None


def plot_tf(power, itc, t_window=None, baseline=None, mode='logratio',
            show=True):
    if t_window is None:
        tmin, tmax = power.times[0] + 0.15, power.times[-1] - 0.15
    else:
        tmin, tmax = t_window

    if isinstance(power, mne.time_frequency.AverageTFR):
        pow_fig = power.plot(picks=None, baseline=baseline, mode=mode,
                             tmin=tmin, tmax=tmax, cmap='RdBu_r', dB=False,
                             show=show)
        itc_fig = None
        if itc is not None:
            itc_fig = itc.plot(picks=None, baseline=baseline, mode=mode,
                               cmap='RdBu_r', tmin=tmin, tmax=tmax, dB=False,
                               show=show)

        return pow_fig, itc_fig

    if isinstance(power, mne.time_frequency.EpochsTFR):
        freq_band = {'alpha': (8, 13), 'beta': (12, 35), 'gamma': (35, 60),
                     'high gamma': (50, 120)}
        power.crop(power.times[0] + 0.15, power.times[-1] - 0.15)
        f_epo = []
        for f in freq_band.keys():
            idx = np.where(np.logical_and(power.freqs > freq_band[f][0],
                                          power.freqs < freq_band[f][1]))

            f_data = power.data.squeeze()[:, idx[0], :].mean(1)
            # m = f_data.mean(1, keepdims=True)
            # f_data = (f_data - m) / m

            f_epo.append(f_data)
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(power.ch_names[0])
        for fi, ei, fn in zip(product(range(2), range(2)),
                              f_epo,
                              freq_band.keys()):
            ax = axs[fi]

            vmin = np.percentile(ei, .1)
            vmax = np.percentile(ei, 99.9)
            times = power.times
            n_epo = np.array(range(power.data.shape[0]))

            ax.set_title(fn)
            pcm = ax.pcolormesh(times, n_epo, ei,
                                vmin=vmin, vmax=vmax,
                                cmap='viridis')
            fig.colorbar(pcm, ax=ax)
            # lp = ax.plot(times, ei.std(0), 'w')
            lp = ax.plot(times.min() +
                         np.std((ei - ei.mean(1, keepdims=True)) /
                                ei.mean(1, keepdims=True), 1), n_epo, 'w')
        if show:
            plt.show()

        return fig


def tf_diff(tf_1, tf_2, baseline=None, mode='logratio', tmin=None, tmax=None,
            cmap='viridis', show=True):
    assert np.all(tf_1.times == tf_2.times), AssertionError('Times must be '
                                                            'the same')
    assert np.all(tf_1.freqs == tf_2.freqs), AssertionError('Freqs must be '
                                                            'the same')
    assert tf_1.nave == tf_2.nave, AssertionError('Number of average TFRs'
                                                  'must be the same')
    assert tf_1.method == tf_2.method, AssertionError('Method must be '
                                                      'the same')
    diff = tf_1.copy().data - tf_2.copy().data

    pow_diff = mne.time_frequency.AverageTFR(info=tf_1.info,
                                             data=diff,
                                             times=tf_1.times,
                                             freqs=tf_1.freqs,
                                             nave=tf_1.nave,
                                             comment=None,
                                             method=tf_1.method)
    pow_diff.plot(baseline=baseline, mode=mode, tmin=tmin, tmax=tmax,
                  cmap=cmap, show=show)
    return pow_diff



if __name__ == '__main__':
    import os
    from lfp_causal.epochs import concatenate_epochs

    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_off'

    epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)

    files = []
    for file in os.listdir(epo_dir):
        # file = '0975_{}_epo.fif'.format(event)
        if file.endswith('{0}_epo.fif'.format(event)):
            session = file.replace('_{0}_epo.fif'.format(event), '')
            epo_fname = os.path.join(epo_dir, file)

            epo = reject_bad_epochs(epo_fname, monkey, condition, session)[0]

            epochs_tf_analysis(epo_fname, baseline=(-1.5, -.5), avg=True)
            epochs_tf_analysis(epo_fname, baseline=(-1.5, -.5), avg=False)

    #         files.append(epo_fname)
    # epochs = concatenate_epochs(files)
    # epochs_tf_analysis(epochs, baseline=(-2.1, -1.9))
