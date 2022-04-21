import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')

from research.get_dirs import get_dirs
from lfp_causal.IO import read_sector
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo


def plot_regs(rfname, reg_name, log_bad=None, bad_trials=None, avg=True,
              get_cov=True):
    if isinstance(reg_name, str):
        reg_name = [reg_name]
    if log_bad is not None:
        assert len(rfname) == len(log_bad)
    elif log_bad is None:
        log_bad = [[]] * len(rfname)
    if bad_trials is not None:
        assert len(rfname) == len(bad_trials)
    elif bad_trials is None:
        bad_trials = [[]] * len(rfname)

    assert len(log_bad) == len(bad_trials)

    d = {k: [] for k in reg_name}
    for r in reg_name:
        regs = np.full((len(rfname), 25), np.nan)
        for (i, f), lb, bt in zip(enumerate(rfname), log_bad, bad_trials):
            xls = pd.read_excel(f)
            reg = xls[r].values

            if len(lb) != 0:
                reg = np.delete(reg, lb)
            if len(bt) != 0:
                reg = np.delete(reg, bt)

            reg = reg[:25]

            regs[i, :len(reg)] = reg

        if avg is True:
            d[r] = np.nanmean(regs, axis=0)
        elif avg is False:
            d[r] = regs.T

    n_col = 5
    # n_row = len(reg_name) // n_col

    if len(reg_name) % n_col == 0:
        n_row = len(reg_name) // n_col
        fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4),
                                sharex=True)
    elif len(reg_name) % n_col != 0:
        n_row = (len(reg_name) // n_col) + 1
        fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4),
                                sharex=True)

    if len(reg_name) == 1:
        axs.plot(d[r])
        axs.set_title(r, loc='left')

    elif len(reg_name) > 1:
        for ax, rn in zip(axs.flatten(), reg_name):
            ax.plot(d[rn])
            ax.set_title(rn, loc='left')
        # c = 0
        # for _r in range(n_row):
        #     for _c in range(n_col):
        #         if c < len(d):
        #             axs[_r, _c].plot(d[reg_name[c]])
        #             axs[_r, _c].set_title(reg_name[c], loc='left')
        #         else:
        #             pass
        #         c += 1
    # for _r in range(n_row):
    #     for _c in range(n_col):
    #     elif len(reg_name) > 1:
    #         axs[i].plot(d[r])
    #         axs[i].set_title(r, loc='left')
    if len(reg_name) > 1:
        if axs.ndim > 1:
            _axs = len(reg_name) % n_col
            for _i in range(_axs, n_col):
                fig.delaxes(axs[n_row - 1, _i])

    # plt.plot(d[r])
    plt.show()

    if get_cov == True:
        if avg == False:
            for r in reg_name:
                d[r] = np.nanmean(regs, axis=0)
        df = pd.DataFrame.from_dict(d)
        # cov = df.cov()
        corr = df.corr()
        sns.heatmap(corr, vmax=1, vmin=-1, annot=True, cbar=True)
        # fig, ax = plt.subplots(1, 1)
        # ax.pcolormesh(cov)
        # ax.pcolormesh(corr, cmap='RdBu_r')
        # plt.colorbar()
        plt.show()


if __name__ == '__main__':
    monkeys = ['freddie', 'teddy']
    conditions = ['easy', 'hard']
    event = 'trig_off'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']

    rej_files = []
    rej_files += ['1204', '1217', '1231', '0944',  # Bad sessions
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
        all_conds = []
        log_bads = []
        bad_epo = []

        for condition in conditions:

            dirs = get_dirs('local', 'lfp_causal')
            # directory = dirs['pow'].format(monkey, condition)
            epo_dir = dirs['epo'].format(monkey, condition)
            regr_dir = dirs['reg'].format(monkey, condition)
            rec_info = op.join(dirs['ep_cnds'].format(monkey, condition),
                               'files_info.xlsx')

            for sect in sectors:
                fid = read_sector(rec_info, sect)
                fid = fid[fid['quality'] <= 3]

                for fs in fid['file']:
                    rname = op.join(regr_dir, '{0}.xlsx'.format(fs))
                    if fs not in rej_files:
                        all_conds.append(rname)

                        fname_epo = op.join(epo_dir,
                                            '{0}_{1}_epo.fif'.format(fs,
                                                                     event))
                        lb = get_log_bad_epo(fname_epo)
                        log_bads.append(lb)
                        be = get_ch_bad_epo(monkey, condition, fs,
                                            fname_info=rec_info)
                        bad_epo.append(be)
        # regres = ['Reward', 'q_rpe', 'q_absrpe', 'RT', 'MT', 'q_dP', 'Actions']
        regres = ['q_rpe', 'q_absrpe', 'RT', 'MT', 'Actions']

        plot_regs(all_conds, regres, log_bad=None, bad_trials=None, avg=True)
        # plot_regs(all_conds, regres, log_bad=log_bads, bad_trials=bad_epo, avg=True)
