import numpy as np
import mne
from scipy.stats import sem
from scipy.stats import ttest_ind
import pandas as pd
import os
import os.path as op
import matplotlib.pyplot as plt
from lfp_causal.IO import read_bad_epochs, read_sector, read_session
from lfp_causal.evoked import plot_evoked
from lfp_causal.tf_analysis import epochs_tf_analysis
from lfp_causal.visu import DraggableColorbar


def plot_rew_evo(epochs, sessions, eve_dir, crop=None, freqs=None,
                 colors=('k', 'r'), show=True):

    fig, ax = plt.subplots(1, 1)
    fig_avg, ax_avg = plt.subplots(1, 1)

    all_p_evo = []
    all_n_evo = []
    pr_trials, nr_trials = 0, 0
    for epo, ses in zip(epochs, sessions):
        if isinstance(epo, str):
            epo = mne.read_epochs(epo, preload=True)

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

        eve = mne.read_events(op.join(eve_dir, '{0}_eve.fif'.format(ses)))
        eve = eve[eve[:, -1] == 20]

        _b = []
        for i, d in enumerate(epo.drop_log):
            if d != ():
                _b.append(i)
        eve = np.delete(eve, _b, 0)

        if 'mov_on' in epo.filename:
            monkey = epo.filename.split('/')[-4]
            condition = epo.filename.split('/')[-3]
            csv_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/' \
                      'lfp_causal/{0}/{1}/t_events'.format(monkey, condition)
            csv = pd.read_csv(op.join(csv_dir, '{0}.csv'.format(ses)))
            movements = csv['mov_on'].values
            movements = np.delete(movements, _b)
            eve = eve[np.isfinite(movements), :]

        p_evo = epo.copy()[eve[:, 1] == 1].average()
        n_evo = epo.copy()[eve[:, 1] == 0].average()
        pr_trials += len(epo.copy()[eve[:, 1] == 1])
        nr_trials += len(epo.copy()[eve[:, 1] == 0])

        if type(all_p_evo) == list:
            all_p_evo = p_evo.data
            all_n_evo = n_evo.data
        else:
            all_p_evo = np.vstack((all_p_evo, p_evo.data))
            all_n_evo = np.vstack((all_n_evo, n_evo.data))

        fig = plot_evoked(p_evo, fig=fig, color=colors[0],
                          label=ses, vh_lines=False, show=False)
        fig = plot_evoked(n_evo, fig=fig, color=colors[1],
                          label=ses, vh_lines=False, show=False)

    ax_avg.plot(epo.times, all_p_evo.mean(0), color=colors[0],
                label='reward, {0} trials'.format(pr_trials))
    ax_avg.plot(epo.times, all_n_evo.mean(0), color=colors[1],
                label='no reward, {0} trials'.format(nr_trials))
    ax_avg.fill_between(epo.times, all_p_evo.mean(0) + sem(all_p_evo, 0),
                        all_p_evo.mean(0) - sem(all_p_evo, 0),
                        color=colors[0], alpha=.2)
    ax_avg.fill_between(epo.times, all_n_evo.mean(0) + sem(all_n_evo, 0),
                        all_n_evo.mean(0) - sem(all_n_evo, 0),
                        color=colors[1], alpha=.2)
    ax_avg.axvline(0, linestyle='--', color='k', linewidth=.8)
    ax_avg.axhline(0, linestyle='-', color='k', linewidth=.8)

    plt.legend()
    if show:
        plt.show()

    return fig, fig_avg


def plot_rew_power(epochs, freqs=(7, 30), n_cycles=None, bline=None):
    t_win = (-.5, 1.)
    p_epo, n_epo = [], []
    for epo in epochs:
        if isinstance(epo, str):
            epo = mne.read_epochs(epo, preload=True)
        epo.rename_channels({epo.ch_names[0]: 'LFP'})

        ses = epo.filename.split('/')[-1].split('_')[0]

        eve = mne.read_events(op.join(eve_dir, '{0}_eve.fif'.format(ses)))
        eve = eve[eve[:, -1] == 20]

        _b = []
        for i, d in enumerate(epo.drop_log):
            if d != ():
                _b.append(i)

        if 'mov_on' in epo.filename:
            monkey = epo.filename.split('/')[-4]
            condition = epo.filename.split('/')[-3]
            csv_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/' \
                      'lfp_causal/{0}/{1}/t_events'.format(monkey, condition)
            csv = pd.read_csv(op.join(csv_dir, '{0}.csv'.format(ses)))
            movements = csv['mov_on'].values
            movements = np.delete(movements, _b)
            eve = eve[np.isfinite(movements), :]

        eve = np.delete(eve, _b, 0)
        p_epo.append(epo.copy()[eve[:, 1] == 1])
        n_epo.append(epo.copy()[eve[:, 1] == 0])

    p_epo = mne.concatenate_epochs(p_epo)
    n_epo = mne.concatenate_epochs(n_epo)

    p_pow, p_itc, (pp_fig, pi_fig) = epochs_tf_analysis(p_epo,
                                                        t_win=t_win,
                                                        freqs=freqs,
                                                        n_cycles=n_cycles,
                                                        avg=True,
                                                        baseline=bline,
                                                        show=False)
    n_pow, n_itc, (np_fig, ni_fig) = epochs_tf_analysis(n_epo,
                                                        t_win=t_win,
                                                        freqs=freqs,
                                                        n_cycles=n_cycles,
                                                        avg=True,
                                                        baseline=bline,
                                                        show=False)

    avg_pow, ax_avg_pow = plt.subplots(1, 1)
    for p in [p_pow, n_pow]:
        d_mask = np.logical_and(p.times > t_win[0], p.times < t_win[1])
        data = p.data.squeeze()[:, d_mask]
        times = p.times[d_mask]
        b_mask = np.logical_and(p.times > bline[0], p.times < bline[1])
        bl = p.data.squeeze()[:, b_mask].mean(1, keepdims=True)
        # data = ((data / bl) - bl).mean(0)
        data /= bl
        data = np.log10(data).mean(0)
        ax_avg_pow.plot(times, data)
        ax_avg_pow.axvline(0, linestyle='--', color='k')

    diff_pow, ax_diff_pow = plt.subplots(1, 1)
    im_diff_pow = ax_diff_pow.pcolormesh(p_pow.times[d_mask], p_pow.freqs,
                                         p_pow.data.squeeze()[:, d_mask] -
                                         n_pow.data.squeeze()[:, d_mask],
                                         cmap='RdBu_r')
    diff_cbar = DraggableColorbar(diff_pow.colorbar(
        mappable=im_diff_pow, ax=ax_diff_pow), im_diff_pow)
    plt.show()
    return


def comparative_rew_plot(epochs, sessions, freqs=None, crop=None):
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
               '{0}/{1}/recording_info.xlsx'.format(monkey, condition)
    eve_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/eve'.format(monkey, condition)

    fig = plt.figure()
    gs = fig.add_gridspec(2, 3)

    ax_asso = fig.add_subplot(gs[0, 0]).set_title('Associative')
    ax_moto = fig.add_subplot(gs[0, 1]).set_title('Motor')
    ax_limb = fig.add_subplot(gs[0, 2]).set_title('Limbic')
    ax_comp = fig.add_subplot(gs[1, :])
    colors = [('#C0392B', '#27AE60'), ('#7D3C98', '#F1C40F'),
              ('#2471A3', '#E67E22')]

    epo_div = [[], [], []]
    ses_div = [[], [], []]
    # for sec in sectors:
    for epo, ses in zip(epochs, sessions):
        sec = read_session(rec_info, ses)['sector'].item()
        for s in range(len(sectors)):
            if sec == sectors[s]:
                epo_div[s].append(epo)
                ses_div[s].append(ses)

    for s, c in zip(range(len(sectors)), colors):
        fig_s, fig_s_avg = plot_rew_evo(epo_div[s], ses_div[s], eve_dir,
                                        freqs=freqs, crop=crop,
                                        colors=(c[0], c[1]), show=False)
        for ax_s in fig_s.axes:
            for ln_s in ax_s.lines[:-2]:
                fig.axes[s].plot(*ln_s.get_data(), color=ln_s.get_c(),
                                 label=ln_s.get_label())
        for ax_s_avg in fig_s_avg.axes:
            for ln_s_avg in ax_s_avg.lines[:-2]:
                fig.axes[-1].plot(*ln_s_avg.get_data(), color=ln_s_avg.get_c(),
                                 label=ln_s_avg.get_label())

    for ax in fig.axes:
        ax.axvline(0, linestyle='--', color='k', linewidth=.8)
        ax.axhline(0, linestyle='-', color='k', linewidth=.8)
    plt.legend()
    plt.show(fig)


def raw_t_test(epochs, sessions, crop=None, freqs=None):

    all_p_evo = []
    all_n_evo = []
    for epo, ses in zip(epochs, sessions):
        if isinstance(epo, str):
            epo = mne.read_epochs(epo, preload=True)

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

        eve = mne.read_events(op.join(eve_dir, '{0}_eve.fif'.format(ses)))
        eve = eve[eve[:, -1] == 20]

        _b = []
        for i, d in enumerate(epo.drop_log):
            if d != ():
                _b.append(i)
        eve = np.delete(eve, _b, 0)

        # if 'mov_on' in epo.filename:
        monkey = epo.filename.split('/')[-4]
        condition = epo.filename.split('/')[-3]
        csv_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/' \
                  'lfp_causal/{0}/{1}/t_events'.format(monkey, condition)
        csv = pd.read_csv(op.join(csv_dir, '{0}.csv'.format(ses)))
        movements = csv['mov_on'].values
        movements = np.delete(movements, _b)
        eve = eve[np.isfinite(movements), :]
        if 'mov_on' not in epo.filename:
            epo = epo[np.isfinite(movements)]

        p_evo = epo.copy()[eve[:, 1] == 1]
        n_evo = epo.copy()[eve[:, 1] == 0]

        # p_evo = epo.copy()[eve[:, 1] == 1].average()
        # n_evo = epo.copy()[eve[:, 1] == 0].average()
        # pr_trials += len(epo.copy()[eve[:, 1] == 1])
        # nr_trials += len(epo.copy()[eve[:, 1] == 0])

        if type(all_p_evo) == list:
            all_p_evo = p_evo._data.squeeze()
            all_n_evo = n_evo._data.squeeze()
        else:
            all_p_evo = np.vstack((all_p_evo, p_evo._data.squeeze()))
            all_n_evo = np.vstack((all_n_evo, n_evo._data.squeeze()))

    all_p_evo = np.array(all_p_evo)
    all_n_evo = np.array(all_n_evo)
    times = epo.times
    t_stat, t_vals = ttest_ind(all_p_evo, all_n_evo, axis=0, equal_var=False)

    t_all_p = all_p_evo.mean(0).copy()
    t_all_p[t_vals >= 0.05] = np.nan
    t_all_n = all_n_evo.mean(0).copy()
    t_all_n[t_vals >= 0.05] = np.nan

    hbar = t_all_p.copy()
    hbar[np.isfinite(hbar)] = t_all_p[np.isfinite(t_all_p)].min() - 0.005

    fig, ax = plt.subplots()
    ax.plot(times, all_p_evo.mean(0), color='g', linewidth=2.)
    ax.plot(times, all_n_evo.mean(0), color='r', linewidth=2.)
    ax.plot(times, t_all_p, color='g', linewidth=5., alpha=0.7)
    ax.plot(times, t_all_n, color='r', linewidth=5., alpha=0.7)
    ax.axvline(x=0., ymin=0, ymax=1, color='k', linestyle='--')
    ax.plot(times, hbar, 'k')


    plt.show()
    return


def tfr_t_test(epochs, sessions, crop=None, freqs=None):

    all_p_evo = []
    all_n_evo = []
    for epo, ses in zip(epochs, sessions):
        if isinstance(epo, str):
            epo = mne.read_epochs(epo, preload=True)

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

        eve = mne.read_events(op.join(eve_dir, '{0}_eve.fif'.format(ses)))
        eve = eve[eve[:, -1] == 20]

        _b = []
        for i, d in enumerate(epo.drop_log):
            if d != ():
                _b.append(i)
        eve = np.delete(eve, _b, 0)

        # if 'mov_on' in epo.filename:
        monkey = epo.filename.split('/')[-4]
        condition = epo.filename.split('/')[-3]
        csv_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/' \
                  'lfp_causal/{0}/{1}/t_events'.format(monkey, condition)
        csv = pd.read_csv(op.join(csv_dir, '{0}.csv'.format(ses)))
        movements = csv['mov_on'].values
        movements = np.delete(movements, _b)
        eve = eve[np.isfinite(movements), :]
        if 'mov_on' not in epo.filename:
            epo = epo[np.isfinite(movements)]

        p_evo = epo.copy()[eve[:, 1] == 1]
        n_evo = epo.copy()[eve[:, 1] == 0]

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
    # sectors = ['associative striatum']
    # sectors = ['limbic striatum']

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
            # file = '0832'
            if file not in rej_ses:
                fname_epo = op.join(epo_dir,
                                         '{0}_{1}_epo.fif'.format(file, event))

                if op.exists(fname_epo):
                    epo_fname.append(fname_epo)
                    ses_n.append(file)

    # fig, fig_avg = plot_rew_evo(epo_fname, ses_n, eve_dir, crop=(-1.5, 1.5), colors=('b', 'g'))
    # plot_rew_power(epo_fname, freqs=(1, 100), n_cycles=None, bline=(-2., -1.))
    #
    # comparative_rew_plot(epo_fname, ses_n, freqs=(30, 120), crop=(-.5, 1.5))

    # rew_t_test(epo_fname, ses_n, freqs=(10, 30), crop=(-2., 1.5))
    # raw_t_test(epo_fname, ses_n, freqs=(10, 30), crop=(-2, 1.5))

    tfr_t_test(epo_fname, ses_n, freqs=(10, 30), crop=(-2, 1.5))
