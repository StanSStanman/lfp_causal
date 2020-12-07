import mne
import os.path as op
import pandas as pd
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from lfp_causal.evoked import plot_evoked
from lfp_causal.IO import read_sector


def plot_mov_evo(epochs, csv_dir):

    fig, ax = plt.subplots(1, 1)
    fig_avg, ax_avg = plt.subplots(1, 1)

    all_mov = []
    all_nomov = []
    ntr_mov, ntr_nomov = 0, 0
    for epo in epochs:
        if isinstance(epo, str):
            epo = mne.read_epochs(epo, preload=True)

        ses = epo.filename.split('/')[-1].split('_')[0]

        epo.crop(-1., 1.)

        _b = []
        for i, d in enumerate(epo.drop_log):
            if d != ():
                _b.append(i)

        csv = pd.read_csv(op.join(csv_dir, '{0}.csv'.format(ses)))
        movements = csv['mov_on'].values
        movements = np.delete(movements, _b)

        mov_evo = epo.copy()[np.isfinite(movements)].average()
        nomov_evo = epo.copy()[np.isnan(movements)].average()

        ntr_mov += np.sum(np.isfinite(movements))
        ntr_nomov += np.sum(np.isnan(movements))

        if type(all_mov) == list:
            all_mov = mov_evo.data
            all_nomov = nomov_evo.data
        else:
            all_mov = np.vstack((all_mov, mov_evo.data))
            all_nomov = np.vstack((all_nomov, nomov_evo.data))

        fig = plot_evoked(mov_evo, fig=fig, color='k',
                          label=ses, show=False)
        fig = plot_evoked(nomov_evo, fig=fig, color='r',
                          label=ses, show=False)

    ax_avg.plot(epo.times, all_mov.mean(0), color='k',
                label='movement, {0} trials'.format(ntr_mov))
    ax_avg.plot(epo.times, np.nanmean(all_nomov, 0), color='r',
                label='no movement, {0} trials'.format(ntr_nomov))
    ax_avg.fill_between(epo.times, all_mov.mean(0) + sem(all_mov, 0),
                        all_mov.mean(0) - sem(all_mov, 0),
                        color='k', alpha=.2)
    ax_avg.fill_between(epo.times,
                        np.nanmean(all_nomov, 0) +
                        sem(all_nomov, 0, nan_policy='omit'),
                        np.nanmean(all_nomov, 0) -
                        sem(all_nomov, 0, nan_policy='omit'),
                        color='r', alpha=.2)
    ax_avg.axvline(0, linestyle='--', color='k', linewidth=.8)
    ax_avg.axhline(0, linestyle='-', color='k', linewidth=.8)

    plt.legend()
    plt.show()

    return fig, fig_avg


if __name__ == '__main__':

    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_on'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    sectors = ['motor striatum']
    # sectors = ['associative striatum']
    # sectors = ['limbic striatum']

    rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
               '{0}/{1}/recording_info.xlsx'.format(monkey, condition)
    epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)
    csv_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/' \
              'lfp_causal/{0}/{1}/t_events'.format(monkey, condition)

    rej_ses = ['0831', '0837']
    rej_ses = []

    epo_fname = []
    ses_n = []
    for sect in sectors:
        fid = read_sector(rec_info, sect)

        for file in fid['file']:
            # file = '1398'
            if file not in rej_ses:
                fname_epo = op.join(epo_dir,
                                         '{0}_{1}_epo.fif'.format(file, event))

                if op.exists(fname_epo):
                    epo_fname.append(fname_epo)
                    ses_n.append(file)

    fig = plot_mov_evo(epo_fname, csv_dir)
