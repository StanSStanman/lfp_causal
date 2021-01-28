import os
import os.path as op
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from lfp_causal.evoked import epo_to_evo
from lfp_causal.IO import read_sector, read_session


def evoked_covariance(evokeds, sessions, labels, cmap='plasma'):
    cov = np.cov(evokeds)

    assert cov.shape[0] == len(sessions) and cov.shape[0] == len(labels), \
        IndexError('Covariance matrix dimension and sessions/labels '
                   'number do not correspond')

    df_cov = pd.DataFrame(cov, index=labels, columns=sessions)

    palette = sns.color_palette(cmap, len(set(labels)))
    colors = np.empty((len(labels), 3))
    for l, p in zip(set(labels), palette):
        colors[np.array(labels) == l] = p

    fig = sns.clustermap(df_cov,  method='centroid', metric='euclidean',
                         row_colors=colors, col_colors=colors,
                         cmap=cmap, cbar_pos=(.1, .2, .03, .5),
                         xticklabels=True, yticklabels=True,
                         center=0.)
    fig.ax_row_dendrogram.set_visible(False)
    plt.show()

    return


def freqs_covariance(powers, freqs, times, cmap='plasma'):

    ftp = []
    for p in powers:
        if isinstance(p, str):
            p = xr.load_dataset(p)
        print('Processing session', list(p.keys())[0])
        p = p.mean('trials')
        p = p.loc[dict(times=slice(times[0], times[1]))]

        _ftp = []
        for f in freqs:
            p_f = p.loc[dict(freqs=slice(f[0], f[1]))]
            p_f = p_f.mean('freqs')
            _ftp.append(np.array(p_f.to_array()))
        _ftp = np.vstack(tuple(_ftp))

        if isinstance(ftp, list):
            ftp = _ftp
        else:
            ftp = (ftp + _ftp) / 2

    times = p_f.times.data
    labels = [str(_f) for _f in freqs]

    t_step = 3
    t_win = 7
    cycles = round((len(times - t_win)) / t_step)
    _st, _et = 0, 0
    cov_ftp = []
    _times = np.zeros(cycles)
    for c in range(cycles):
        _et = _st + t_win
        cov_ftp.append(np.cov(ftp[:, _st:_et]))
        _times[c] = times[_st:_et].mean()
        _st += t_step
    cov_ftp = np.dstack(tuple(cov_ftp))
    idxs = np.triu_indices(len(freqs), k=1)
    cov_ftp = cov_ftp[idxs[0], idxs[1], :]

    fig_cov, ax_cov = plt.subplots(1, 1)
    for l, c in zip(combinations(labels, 2), cov_ftp):

        ax_cov.plot(_times, c, label=l)
        # ax_cov.set_xticklabels(labels, rotation=90)
        # ax_cov.set_yticklabels(labels)
    plt.legend()
    plt.show()

    # df_cov = pd.DataFrame(cov_ftp, index=labels, columns=labels)
    #
    # # palette = sns.color_palette(cmap, len(labels))
    # # colors = np.empty((len(labels), ))
    # # for l, p in zip(labels, palette):
    # #     colors[np.array(labels) == l] = p
    #
    # palette = sns.color_palette(cmap, len(set(labels)))
    # colors = np.empty((len(labels), 3))
    # for l, p in zip(set(labels), palette):
    #     colors[np.array(labels) == l] = p
    #
    # fig = sns.clustermap(df_cov,  method='centroid', metric='euclidean',
    #                      row_colors=colors, col_colors=colors,
    #                      cmap=cmap, cbar_pos=(.1, .2, .03, .5),
    #                      xticklabels=True, yticklabels=True,
    #                      center=0.)
    # fig.ax_row_dendrogram.set_visible(False)
    # plt.show()



if __name__ == '__main__':

    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_off'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']

    rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
               '{0}/{1}/files_info.xlsx'.format(monkey, condition)
    epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)
    pow_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
              '{0}/{1}/pow'.format(monkey, condition)
    ###########################################################################

    # data_m = []
    # ses_n = []
    # sec_name = []
    # for sect in sectors:
    #     fid = read_sector(rec_info, sect)
    #
    #     for file in fid['file']:
    #         fname_epo = os.path.join(epo_dir,
    #                                  '{0}_{1}_epo.fif'.format(file, event))
    #
    #         if os.path.exists(fname_epo):
    #             evo = epo_to_evo(fname_epo)
    #             evo.crop(-.5, .5)
    #             data = evo.data
    #
    #             if isinstance(data_m, list):
    #                 data_m = data
    #             else:
    #                 data_m = np.vstack((data_m, data))
    #
    #             ses_n.append(file)
    #             sec_name.append(sect)
    #
    # evoked_covariance(data_m, ses_n, sec_name)
    ###########################################################################

    pow_name = '{0}_pow_5_120.nc'.format(event)
    freqs = [(8, 15), (15, 30), (25, 45), (40, 70), (60, 120)]
    # freqs = [(30, 50), (35, 55)]
    time = (-1.8, 1.3)

    d = {s: [] for s in sectors}

    # powers = []
    for pdr in os.listdir(pow_dir)[:-1]:
        _s = read_session(rec_info, pdr)['sector'].values[0]
        # powers.append(op.join(pow_dir, pdr, pow_name))
        d[_s].append(op.join(pow_dir, pdr, pow_name))

    for sec in sectors:
        freqs_covariance(d[sec], freqs, time, cmap='plasma')
