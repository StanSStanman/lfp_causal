import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lfp_causal.evoked import epo_to_evo
from lfp_causal.IO import read_sector


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

    fig = sns.clustermap(df_cov, row_colors=colors, col_colors=colors,
                         cmap=cmap, cbar_pos=(.1, .2, .03, .5),
                         xticklabels=True, yticklabels=True,
                         center=0.)
    fig.ax_row_dendrogram.set_visible(False)
    plt.show()

    return


if __name__ == '__main__':

    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_off'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']

    rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
               '{0}/{1}/recording_info.xlsx'.format(monkey, condition)
    epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)

    data_m = []
    ses_n = []
    sec_name = []
    for sect in sectors:
        fid = read_sector(rec_info, sect)

        for file in fid['file']:
            fname_epo = os.path.join(epo_dir,
                                     '{0}_{1}_epo.fif'.format(file, event))

            if os.path.exists(fname_epo):
                evo = epo_to_evo(fname_epo)
                evo.crop(-.1, 1)
                data = evo.data

                if isinstance(data_m, list):
                    data_m = data
                else:
                    data_m = np.vstack((data_m, data))

                ses_n.append(file)
                sec_name.append(sect)

    evoked_covariance(data_m, ses_n, sec_name)
