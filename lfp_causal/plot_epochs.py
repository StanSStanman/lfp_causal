import xarray as xr
import os
import os.path as op
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research.get_dirs import get_dirs
from lfp_causal.compute_power import normalize_power
from lfp_causal.IO import read_sector
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo
from lfp_causal.profiling import (RepeatedTimer, memory_usage, cpu_usage)


def plot_avg_epo(epochs, bads):
    avg_epo = None
    # if blines is None:
    #     blines = range(len(powers))
    for p, b in zip(epochs, bads):
        print('Loading', p)
        epo = mne.read_epochs(p)
        epo = epo.drop(b)
        epo = epo.average()
        # pow = pow.loc[dict(times=slice(-.15, .8))]
        # # pow = pow.loc[dict(freqs=slice(70, 120))]
        # pow = pow.to_array().squeeze().mean('trials')

        if avg_epo is None:
            avg_epo = epo.data.T
        else:
            avg_epo = np.hstack((avg_epo, epo.data.T))

    # avg_epo.plot(cmap='viridis')
    fig, ax = plt.subplots(1, 1)
    cm = ax.plot(epo.times, avg_epo)
    # plt.colorbar(cm)
    plt.show()
    return


if __name__ == '__main__':
    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_off'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    # sectors = ['limbic striatum']

    dirs = get_dirs('local', 'lfp_causal')
    directory = dirs['pow'].format(monkey, condition)
    epo_dir = dirs['epo'].format(monkey, condition)
    regr_dir = dirs['reg'].format(monkey, condition)
    rec_info = op.join(dirs['ep_cnds'].format(monkey, condition),
                       'files_info.xlsx')

    # rej_files = ['0845', '0847', '0873', '0911', '0939', '0945', '0946',
    #              '0948', '0951', '0956', '0963', '0968', '1024', '1036',
    #              '1038', '1056', '1204', '1217'] + \
    #             ['0944', '0967', '0969', '0967', '0970', '0971', '0985',
    #              '1139', '1145', '1515', '1701'] + \
    #             ['0814', '0831', '0850', '0866', '0923', '0941', '1135',
    #              '1138', '1234', '1235', '1248', '1299', '1302', '1397',
    #              '1398', '1514', '1699', '1283']
    # acc_files = ['0947', '0949', '0952']
    rej_files = ['1204', '1217', '1231', '0944', # Bad sessions
                 '0845', '0847', '0939', '0946', '0963', '1036', '1231',
                 '1233', '1234', '1514', '1699',
                 '0940', '0944', '0964', '0967', '0969', '0970', '0971',
                 '0977', '0985', '1280']

    for sect in sectors:
        fid = read_sector(rec_info, sect)
        fid = fid[fid['quality'] <= 3]
        # fid = fid[fid['neuron_type'] == 'TAN']

        all_files = []
        all_bline = []
        all_conds = []
        log_bads = []
        bad_epo = []
        for fs in fid['file']:
            fname = op.join(epo_dir, '{0}_{1}_epo.fif'.format(fs, event))
            rname = op.join(regr_dir, '{0}.xlsx'.format(fs))
            if op.exists(fname) and fs not in rej_files:
            # if op.exists(fname) and fs in acc_files:
                all_files.append(fname)
                all_conds.append(rname)
                #
                # fname_epo = op.join(epo_dir,
                #                     '{0}_{1}_epo.fif'.format(fs, event))
                # lb = get_log_bad_epo(fname_epo)
                # log_bads.append(lb)
                be = get_ch_bad_epo(monkey, condition, fs,
                                    fname_info=rec_info)
                bad_epo.append(be)
                # plot_avg_tf([fname], [bname])

        plot_avg_epo(all_files, bad_epo)
        # plot_pow_cond(all_files, all_conds, 'Reward', log_bads, bad_epo, norm,
        #               (-.52, -.02), all_bline, c_type='d', bins=2, diff=True,
        #               plot='pcm')