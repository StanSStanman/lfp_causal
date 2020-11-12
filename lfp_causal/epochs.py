import numpy as np
import mne
import pandas as pd


def create_epochs(fname_raw, fname_eve, event, tmin, tmax, bline, fname_out,
                  ch_drop='manual'):
    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    eve = mne.read_events(fname_eve)

    # if raw.info['sfreq'] == 16667.0:
    #     print('Resampling at 12500.0 Hz...')
    #     raw.resample(12500, n_jobs=1)

    events_name = ['cue_on', 'cue_off', 'trig_on', 'mov_on', 'trig_off']
    events_id = [[11], [10], [21], [31], [20]]
    # events_id = [[11], [10], [21], [31], [102, 103, 104]]

    assert event in events_name, AssertionError('Event unknown')

    eve_id = events_id[events_name.index(event)]
    if len(eve_id) == 1:
        eve_id = eve_id[0]
    else:
        eve_id = 20

    eve_dict = {}
    for en, ei in zip(events_name, events_id):
        _ev = np.array(([]))
        for _i in ei:
            if _ev.size == 0:
                _ev = eve[eve[:, -1] == _i]
            else:
                _ev = np.vstack((_ev, eve[eve[:, -1] == _i]))
            if en == 'trig_off':
                _ev[:, -1] = 20
        _ev = _ev.astype(int)
        _ev = _ev[np.argsort(_ev[:, 0])]

        eve_dict[en] = _ev

    spk_ch = [s for s in raw.ch_names if 'spike' in s]
    raw.info['bads'] = spk_ch
    # raw.info['bads'] = ['spikes1', 'spikes2']
    if ch_drop == 'manual':
        raw.plot(scalings={'seeg': 0.5}, block=True)
    elif isinstance(ch_drop, list):
        raw.drop_channels(ch_drop)
    else:
        pass
    # raw.save(raw.filenames[0], overwrite=True)
    raw.notch_filter(np.arange(50, 251, 50),
                     notch_widths=np.arange(50, 251, 50) / 200)
    # raw.notch_filter(freqs=np.arange(50, 251, 50), phase='minimum',
    #                  fir_window='hann', method='fir', n_jobs=2)
    raw.filter(l_freq=1., h_freq=140., n_jobs=-1)


    epo = mne.Epochs(raw, eve_dict[event], eve_id, tmin, tmax, baseline=bline)
    epo.load_data()
    epo.resample(1000., n_jobs=-1)
    epo.save(fname_out, overwrite=True)

    return epo


def visualize_epochs_mne(epo_fname):
    epochs = mne.read_epochs(epo_fname)
    for en in epochs.ch_names:
        if 'LFP' in en:
            fig = epochs.copy().pick(en).plot_image(title=en)

    return fig


def visualize_epochs(epochs, picks=None, block=True, show=True):
    import matplotlib.pyplot as plt
    from lfp_causal.visu import DraggableColorbar
    from scipy.stats import sem

    if isinstance(epochs, str):
        epochs = mne.read_epochs(epochs, preload=True)
    data = epochs.get_data(picks=picks)
    data = data.mean(1)

    times = epochs.times
    n_epo = range(data.shape[0])

    vmin = np.percentile(data, .05)
    vmax = np.percentile(data, 99.95)

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    pcm = axs[0].pcolormesh(times, n_epo, data,
                            vmin=vmin, vmax=vmax,
                            cmap='RdBu_r')
    pcm_vline = axs[0].axvline(0., n_epo[0], n_epo[-1],
                               color='k', linestyle='--',
                               linewidth=0.8)
    cbar = DraggableColorbar(fig.colorbar(pcm, ax=axs[0]), pcm)

    epo_avg = data.mean(0)
    epo_err = sem(data, axis=0)
    lpl = axs[1].plot(times, epo_avg, color='k', linewidth=0.8)
    fbt = axs[1].fill_between(times, epo_avg - epo_err, epo_avg + epo_err,
                              color='k', alpha=0.4)
    lpl_vline = axs[1].axvline(x=0., ymin=-0., ymax=1.,
                               color='k', linestyle='--',
                               linewidth=0.8)

    axs[1].set_xlim([times[0], times[-1]])
    plt.tight_layout()

    if show:
        plt.show(block=block)

    return fig


def concatenate_epochs(epochs_fname):
    all_epo = []
    for ef in epochs_fname:
        epo = mne.read_epochs(ef, preload=True)
        all_epo.append(epo)
    all_epo = mne.concatenate_epochs(all_epo)
    return all_epo


def reject_bad_epochs(epochs, monkey, condition, session, channels=None,
                      drop_bad_ch=True):
    import os.path as op
    csv = pd.read_csv(op.join('/media/jerry/TOSHIBA EXT/data/db_behaviour/',
                              'lfp_causal/{0}/{1}/'.format(monkey, condition),
                              'lfp_bad_trials.csv'), dtype=str)

    if isinstance(epochs, str):
        epochs = mne.read_epochs(epochs, preload=True)

    if drop_bad_ch:
        # epochs.drop_bad()
        epochs.drop_channels(epochs.info['bads'])
        epochs.picks = np.array(range(len(epochs.ch_names)))
        channels = [cn for cn in epochs.ch_names if 'LFP' in cn]

    idx = np.where(csv['session'] == session)
    bads = csv.loc[idx]
    if channels is None:
        channels = []
        for i in bads.keys():
            if 'LFP' in i:
                channels.append(i)
    ch_fn = channels.copy()

    # If the session has a single LFP channel, the column 'LFP1' will be used
    if len(channels) == 1 and channels[0] == 'LFP':
        ch_fn[0] = 'LFP1'

    ch_epo = []
    for ch, fch in zip(channels, ch_fn):
        bad_ep = list(bads[fch])
        bad_ep = [b.split(sep=',') for b in bad_ep][0]
        for i, b in enumerate(bad_ep):
            if b == 'None':
                bad_ep.pop(i)
            else:
                bad_ep[i] = int(b)

        epo = epochs.copy().pick_channels([ch])
        epo.drop(bad_ep)
        ch_epo.append(epo)

    return ch_epo


def auto_drop_chans(fname, session):
    from lfp_causal.IO import read_xls

    xls = read_xls(fname)
    info = xls[xls['file'] == session]

    gc = info['good_channel'].values

    if gc == '0':
        return None
    elif gc == '1':
        return ['LFP2']
    elif gc == '2':
        return ['LFP1']


if __name__ == '__main__':
    import os

    monkey = 'freddie'
    condition = 'easy'
    event = 'cue_off'

    raw_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/raw'.format(monkey, condition)
    eve_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/eve'.format(monkey, condition)
    epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)
    rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
               '{0}/{1}/recording_info.xlsx'.format(monkey, condition)

    files = []
    for file in os.listdir(raw_dir):
        file = '1398_raw.fif'
        if file.endswith('.fif'):
            session = file.replace('_raw.fif', '')
            fname_raw = os.path.join(raw_dir, file)
            fname_eve = os.path.join(eve_dir, file.replace('raw', 'eve'))
            fname_epo = os.path.join(epo_dir,
                                     file.replace('raw',
                                                  '{0}_epo'.format(event)))

            bad_ch = auto_drop_chans(rec_info, session)

            epo = create_epochs(fname_raw, fname_eve,
                                event, -1., 2.,
                                None, fname_epo,
                                ch_drop=bad_ch)

            visualize_epochs(epo, block=True)
            # visualize_epochs(fname_epo)
            # visualize_epochs(fname_epo, ['LFP2'])
