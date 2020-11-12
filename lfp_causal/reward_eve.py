import numpy as np
import mne
from scipy.stats import sem
import os.path as op
import matplotlib.pyplot as plt
from lfp_causal.IO import read_bad_epochs, read_sector
from lfp_causal.evoked import plot_evoked


def plot_rew_evo(epochs, sessions, eve_dir):

    fig, ax = plt.subplots(1, 1)
    fig_avg, ax_avg = plt.subplots(1, 1)

    all_p_evo = []
    all_n_evo = []

    for epo, ses in zip(epochs, sessions):
        if isinstance(epo, str):
            epo = mne.read_epochs(epo, preload=True)

        epo.crop(-.5, 1.)

        eve = mne.read_events(op.join(eve_dir, '{0}_eve.fif'.format(ses)))
        eve = eve[eve[:, -1] == 20]

        _ch = epo.ch_names[0]
        if _ch == 'LFP':
            _ch = 'LFP1'

        _b = []
        for i, d in enumerate(epo.drop_log):
            if d != ():
                _b.append(i)

        eve = np.delete(eve, _b, 0)
        p_evo = epo.copy()[eve[:, 1] == 1].average()
        n_evo = epo.copy()[eve[:, 1] == 0].average()

        if type(all_p_evo) == list:
            all_p_evo = p_evo.data
            all_n_evo = n_evo.data
        else:
            all_p_evo = np.vstack((all_p_evo, p_evo.data))
            all_n_evo = np.vstack((all_n_evo, n_evo.data))

        fig = plot_evoked(p_evo, fig=fig, color='k',
                          label=ses, show=False)
        fig = plot_evoked(n_evo, fig=fig, color='r',
                          label=ses, show=False)

    ax_avg.plot(epo.times, all_p_evo.mean(0), color='k')
    ax_avg.plot(epo.times, all_n_evo.mean(0), color='r')
    ax_avg.fill_between(epo.times, all_p_evo.mean(0) + sem(all_p_evo, 0),
                        all_p_evo.mean(0) - sem(all_p_evo, 0),
                        color='k', alpha=.2)
    ax_avg.fill_between(epo.times, all_n_evo.mean(0) + sem(all_n_evo, 0),
                        all_n_evo.mean(0) - sem(all_p_evo, 0),
                        color='r', alpha=.2)
    ax_avg.axvline(0, linestyle='--', color='k', linewidth=.8)
    ax_avg.axhline(0, linestyle='-', color='k', linewidth=.8)

    plt.show()

    return fig


if __name__ == '__main__':
    import os

    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_off'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    # sectors = ['motor striatum']
    # sectors = ['associative striatum']
    sectors = ['limbic striatum']

    rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
               '{0}/{1}/recording_info.xlsx'.format(monkey, condition)
    epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)
    eve_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/eve'.format(monkey, condition)

    epo_fname = []
    ses_n = []
    for sect in sectors:
        fid = read_sector(rec_info, sect)

        for file in fid['file']:
            # file = '1255'
            fname_epo = os.path.join(epo_dir,
                                     '{0}_{1}_epo.fif'.format(file, event))

            if os.path.exists(fname_epo):
                epo_fname.append(fname_epo)
                ses_n.append(file)

    fig = plot_rew_evo(epo_fname, ses_n, eve_dir)
