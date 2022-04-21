import xarray as xr
import os
import os.path as op
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem
from research.get_dirs import get_dirs
from lfp_causal.compute_power import normalize_power
from lfp_causal.IO import read_sector
from lfp_causal.epochs import visualize_epochs
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
    ax.plot(epo.times, avg_epo.mean(1), color='k', linewidth=2.)
    # plt.colorbar(cm)
    plt.show()
    return


def plot_reg_epo(epochs, reg_fname, bads, regressor, vals, crop=None):
    d = {v: [] for v in vals}
    for p, r, b in zip(epochs, reg_fname, bads):
        print('Loading', p)
        epo = mne.read_epochs(p)
        if crop is not None:
            epo.crop(*crop)
        reg = np.array(pd.read_excel(r)[regressor])
        times = epo.times
        epo = epo.drop(b)
        br = [i for i, dl in enumerate(epo.drop_log) if len(dl) != 0]
        reg = np.delete(reg, br)

        # Cut at 25 trials
        epo = epo[:25]
        reg = reg[:25]

        epo = epo._data.squeeze()
        for k in d.keys():
            _ep = epo[np.where(reg == k)]
            d[k].append(_ep)

    fig, ax = plt.subplots(1, 1)
    dsem = {}
    for k in d.keys():
        d[k] = np.vstack(d[k])
        dsem[k] = sem(d[k], axis=0)
        d[k] = d[k].mean(0)
        ax.plot(times, d[k], linewidth=2., label='Reward={0}'.format(k))
        ax.fill_between(times, d[k] - dsem[k], d[k] + dsem[k], alpha=0.4)
        # ax.plot(epo.times, avg_epo.mean(1), color='k', linewidth=2.)
        # plt.colorbar(cm)
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    monkeys = ['freddie', 'teddy']
    conditions = ['easy', 'hard']
    event = 'trig_off'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    # sectors = ['limbic striatum']

    # acc_files = ['0947', '0949', '0952']
    rej_files = []
    rej_files += ['1204', '1217', '1231', '0944', # Bad sessions
                  '0845', '0847', '0911', '0939', '0946', '0963', '0984',
                  '1036', '1231', '1233', '1234', '1514', '1699',

                  '0940', '0944', '0964', '0967', '0969', '0970', '0971',
                  '0977', '0985', '1037', '1280']
    rej_files += ['0210', '0219', '0221', '0225', '0226', '0227', '0230',
                  '0252', '0268', '0276', '0277', '0279', '0281', '0282',
                  '0283', '0285', '0288', '0290', '0323', '0362', '0365',
                  '0393', '0415', '0447', '0449', '0450', '0456', '0541',
                  '0573', '0622', '0628', '0631', '0643', '0648', '0653',
                  '0660', '0688', '0689', '0690', '0692', '0697', '0706',
                  '0710', '0717', '0718', '0719', '0713', '0726', '0732',

                  '0220', '0223', '0271', '0273', '0278', '0280', '0284',
                  '0289', '0296', '0303', '0363', '0416', '0438', '0448',
                  '0521', '0618', '0656', '0691', '0693', '0698', '0705',
                  '0707', '0711', '0712', '0716', '0720', '0731']

    rej_files += ['0900', '1512', '1555', '1682',
                  '0291', '0368', '0743']

    rej_files += ['0231', '0272', '0274', '0666', '0941', '0855', '0722',
                  '0725', '1397', '1398', '1701']

    for monkey in monkeys:

        all_files = []
        all_bline = []
        all_conds = []
        log_bads = []
        bad_epo = []
        reg_files = []

        for condition in conditions:
            dirs = get_dirs('local', 'lfp_causal')
            directory = dirs['pow'].format(monkey, condition)
            epo_dir = dirs['epo'].format(monkey, condition)
            regr_dir = dirs['reg'].format(monkey, condition)
            rec_info = op.join(dirs['ep_cnds'].format(monkey, condition),
                               'files_info.xlsx')

            for sect in sectors:
                fid = read_sector(rec_info, sect)
                fid = fid[fid['quality'] <= 3]
                # fid = fid[fid['neuron_type'] == 'TAN']

                for fs in fid['file']:
                    fname = op.join(epo_dir, '{0}_{1}_epo.fif'.format(fs, event))
                    rname = op.join(regr_dir, '{0}.xlsx'.format(fs))
                    if op.exists(fname) and fs not in rej_files:
                    # if op.exists(fname) and fs in acc_files:
                        all_files.append(fname)
                        all_conds.append(rname)
                        reg_files.append(rname)
                        #
                        # fname_epo = op.join(epo_dir,
                        #                     '{0}_{1}_epo.fif'.format(fs, event))
                        # lb = get_log_bad_epo(fname_epo)
                        # log_bads.append(lb)
                        be = get_ch_bad_epo(monkey, condition, fs,
                                            fname_info=rec_info)
                        bad_epo.append(be)
                        # plot_avg_tf([fname], [bname])

                # plot_avg_epo(all_files, bad_epo)
                for e, b in zip(all_files, bad_epo):
                    visualize_epochs(e, bads=b, picks=None,
                                     block=True, show=True)
        # plot_reg_epo(all_files, reg_files, bad_epo, 'Reward',
        #              [0, 1], crop=(-.0, .8))

        # plot_pow_cond(all_files, all_conds, 'Reward', log_bads, bad_epo, norm,
        #               (-.52, -.02), all_bline, c_type='d', bins=2, diff=True,
        #               plot='pcm')