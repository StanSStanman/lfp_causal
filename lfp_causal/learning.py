import numpy as np
import mne
import xarray as xr
from scipy.stats import sem
from scipy.stats import ttest_ind
import pandas as pd
import os
import os.path as op
import matplotlib.pyplot as plt
from lfp_causal.IO import read_bad_epochs, read_sector, read_session, read_xls
from lfp_causal.evoked import plot_evoked
from lfp_causal.tf_analysis import epochs_tf_analysis
from lfp_causal.visu import DraggableColorbar


def consec_rep(sequence, n):
    repetitions = np.empty_like(sequence)
    _x = 0
    for i, x in enumerate(sequence):
        if x == 1:
            repetitions[i] = x + _x
            _x += x
        elif x == 0:
            repetitions[i] = 0
            _X = 0
    repetitions[repetitions <= n] = 0
    repetitions[repetitions > n] = 1
    return repetitions


def prep_tfr_data(epochs, sessions, crop=None, freqs=None):
    csv_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/' \
              'lfp_causal/{0}/{1}/t_events'
    fname_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
                 'lfp_causal/{0}/{1}/recording_info.xlsx'

    data = []
    for epo, ses in zip(epochs, sessions):
        if isinstance(epo, str):
            epo = mne.read_epochs(epo, preload=True)

        monkey = epo.filename.split('/')[-4]
        condition = epo.filename.split('/')[-3]
        csv_dir = csv_dir.format(monkey, condition)
        csv = pd.read_csv(op.join(csv_dir, '{0}.csv'.format(ses)))

        if crop is not None:
            assert isinstance(crop, tuple), \
                AssertionError('frequencies should be expressed as a '
                               'tuple(tmin, tmax)')
            epo.crop(crop[0], crop[1])

        if freqs is not None:
            assert isinstance(freqs, tuple), \
                AssertionError('frequencies should be expressed as a '
                               'tuple(fmin, fmax)')
            epo.filter(freqs[0], freqs[1])

        _b = []
        for i, d in enumerate(epo.drop_log):
            if d != ():
                _b.append(i)

        movements = csv['mov_on'].values
        movements = np.delete(movements, _b)
        if 'mov_on' not in epo.filename:
            epo = epo[np.isfinite(movements)]

        d_tfr, _, _ = epochs_tf_analysis(epo, t_win=crop, freqs=freqs,
                                         avg=False, show=False,
                                         baseline=(-2., -1.7))
        plt.close('all')

        if type(data) == list:
            data = d_tfr.data.squeeze().mean(1)
        else:
            data = np.vstack((data, d_tfr.data.squeeze().mean(1)))

    data = xr.DataArray(data,
                        coords=[range(data.shape[0]), d_tfr.times],
                        dims=['trials', 'times'])
    return data


def prep_act_data(epochs, sessions):
    csv_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/' \
              'lfp_causal/{0}/{1}/t_events'
    fname_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
                 'lfp_causal/{0}/{1}/recording_info.xlsx'

    data = []
    for epo, ses in zip(epochs, sessions):
        monkey = epo.split('/')[-4]
        condition = epo.split('/')[-3]
        csv_dir = csv_dir.format(monkey, condition)
        csv = pd.read_csv(op.join(csv_dir, '{0}.csv'.format(ses)))
        infos = read_xls(fname_info.format(monkey, condition))

        correct_pos = infos[infos['file'] == ses]['target_location'].values[0]
        if correct_pos == 'left':
            correct_pos = 104.
        elif correct_pos == 'center':
            correct_pos = 103.
        elif correct_pos == 'right':
            correct_pos = 102.
        actions = csv['button'].values.copy()
        actions[actions != correct_pos] = 0
        actions[actions == correct_pos] = 1

        data.append(actions)
    data = pd.DataFrame(data)
    data = xr.DataArray(data, coords=[sessions, range(data.shape[1])],
                        dims=['sessions', 'trials'])
    return data


def learning_test(epochs, sessions, crop=None, freqs=None):
    csv_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/' \
              'lfp_causal/{0}/{1}/t_events'
    fname_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
                 'lfp_causal/{0}/{1}/recording_info.xlsx'

    all_p_evo = []
    all_n_evo = []
    for epo, ses in zip(epochs, sessions):
        if isinstance(epo, str):
            epo = mne.read_epochs(epo, preload=True)

        monkey = epo.filename.split('/')[-4]
        condition = epo.filename.split('/')[-3]
        csv_dir = csv_dir.format(monkey, condition)
        csv = pd.read_csv(op.join(csv_dir, '{0}.csv'.format(ses)))
        infos = read_xls(fname_info.format(monkey, condition))

        if crop is not None:
            assert isinstance(crop, tuple), \
                AssertionError('frequencies should be expressed as a '
                               'tuple(tmin, tmax)')
            epo.crop(crop[0], crop[1])

        if freqs is not None:
            assert isinstance(freqs, tuple), \
                AssertionError('frequencies should be expressed as a '
                               'tuple(fmin, fmax)')
            epo.filter(freqs[0], freqs[1])

        correct_pos = infos[infos['file'] == ses]['target_location'].values[0]
        if correct_pos == 'left':
            correct_pos = 104.
        elif correct_pos == 'center':
            correct_pos = 103.
        elif correct_pos == 'right':
            correct_pos = 102.
        actions = csv['button'].values.copy()
        actions[actions != correct_pos] = 0
        actions[actions == correct_pos] = 1

        _b = []
        for i, d in enumerate(epo.drop_log):
            if d != ():
                _b.append(i)
        actions = np.delete(actions, _b, 0)

        movements = csv['mov_on'].values
        movements = np.delete(movements, _b)
        actions = actions[np.isfinite(movements)]
        if 'mov_on' not in epo.filename:
            epo = epo[np.isfinite(movements)]

        seq = consec_rep(actions, 4)

        p_evo = epo.copy()[seq == 1]
        n_evo = epo.copy()[seq == 0]

        p_tfr, _, _ = epochs_tf_analysis(p_evo, t_win=crop, freqs=freqs,
                                         avg=False, show=False,
                                         baseline=(-2., -1.7))
        n_tfr, _, _ = epochs_tf_analysis(n_evo, t_win=crop, freqs=freqs,
                                         avg=False, show=False,
                                         baseline=(-2., -1.7))

        # p_evo = epo.copy()[eve[:, 1] == 1].average()
        # n_evo = epo.copy()[eve[:, 1] == 0].average()
        # pr_trials += len(epo.copy()[eve[:, 1] == 1])
        # nr_trials += len(epo.copy()[eve[:, 1] == 0])

        if type(all_p_evo) == list:
            all_p_evo = p_tfr.data.squeeze().mean(1)
            all_n_evo = n_tfr.data.squeeze().mean(1)
        else:
            all_p_evo = np.vstack((all_p_evo, p_tfr.data.squeeze().mean(1)))
            all_n_evo = np.vstack((all_n_evo, n_tfr.data.squeeze().mean(1)))

        plt.close('all')

    all_p_evo = np.array(all_p_evo)
    all_n_evo = np.array(all_n_evo)
    times = p_tfr.times
    t_stat, t_vals = ttest_ind(all_p_evo, all_n_evo, axis=0, equal_var=False)

    t_all_p = all_p_evo.mean(0).copy()
    t_all_p[t_vals >= 0.05] = np.nan
    t_all_n = all_n_evo.mean(0).copy()
    t_all_n[t_vals >= 0.05] = np.nan

    hbar = t_all_p.copy()
    if not np.all(np.isnan(hbar)):
        hbar[np.isfinite(hbar)] = t_all_p[np.isfinite(t_all_p)].min() - 0.005

    fig, ax = plt.subplots()
    ax.plot(times, all_p_evo.mean(0), color='g', linewidth=2.)
    ax.plot(times, all_n_evo.mean(0), color='r', linewidth=2.)
    ax.plot(times, t_all_p, color='g', linewidth=5., alpha=0.7)
    ax.plot(times, t_all_n, color='r', linewidth=5., alpha=0.7)
    ax.axvline(x=0., ymin=0, ymax=1, color='k', linestyle='--')
    ax.plot(times, hbar, 'k')

    plt.show()


if __name__ == '__main__':
    import scipy as sp


    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_off'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    # sectors = ['motor striatum']
    sectors = ['associative striatum']
    sectors = ['limbic striatum']

    rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
               '{0}/{1}/recording_info.xlsx'.format(monkey, condition)
    epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)
    eve_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/eve'.format(monkey, condition)

    rej_ses = ['0831', '0837', '0981', '1043']
    # rej_ses = []

    epo_fname = []
    ses_n = []
    for sect in sectors:
        fid = read_sector(rec_info, sect)

        for file in fid['file']:
            # file = '1397'
            if file not in rej_ses:
                fname_epo = op.join(epo_dir,
                                         '{0}_{1}_epo.fif'.format(file, event))

                if op.exists(fname_epo):
                    epo_fname.append(fname_epo)
                    ses_n.append(file)

    # learning_test(epo_fname, ses_n, freqs=(15, 30), crop=(-1.5, 1.5))
    # prep_tfr_data(epo_fname, ses_n, freqs=(10, 30), crop=(-2, 1.5))
    prep_act_data(epo_fname, ses_n)
