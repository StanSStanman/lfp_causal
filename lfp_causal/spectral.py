import os.path as op
import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as ss
import mne
import matplotlib.pyplot as plt
from research.get_dirs import get_dirs
from lfp_causal.IO import read_sector
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo


def prepare_data(fnames, bads, regs, fbline=None, t_win=(0., 1.), t_bln=None):
    # assert there a bad and regressors files associated to each data file
    assert len(fnames) == len(bads)
    assert len(fnames) == len(regs)
    # assert if there is a baseline file associated to each data file
    if fbline is not None:
        assert isinstance(fbline, list)
        assert len(fnames) == len(fbline)
    elif fbline is None:
        fbline = [None] * len(fnames)

    all_epo = []
    for fn, bl, bd, rg in zip(fnames, fbline, bads, regs):
        # Loading and cutting epochs data
        epo = mne.read_epochs(fn)
        log_bad = get_log_bad_epo(epo)
        epo = epo.drop(bd)
        epo = epo.crop(*t_win)
        times = epo.times
        epo = epo.get_data().squeeze(axis=1)

        # Loading baseline and baseline correction
        if bl is not None:
            bln = mne.read_epochs(bl)
            bln = bln.drop(bd)
            assert len(epo) == len(bln)
            if t_bln is not None:
                bln = bln.crop(*t_bln)
            bln = bln.get_data().squeeze(axis=1)
            epo = ((epo - bln.mean(1, keepdims=True))
                   / bln.mean(1, keepdims=True))

        xls = pd.read_excel(rg,  index_col=0)
        reg = xls['Reward'].values
        if len(log_bad) != 0:
            reg = np.delete(reg, log_bad)
        if len(bd) != 0:
            reg = np.delete(reg, bd)

        assert len(epo) == len(reg)

        # consider only the first 25 trials
        epo = epo[:25, :]
        reg = reg[:25]

        epo = xr.DataArray(epo, coords=[reg, times], dims=['reg', 'times'])

        if isinstance(all_epo, list):
            all_epo = epo
        else:
            all_epo = xr.concat([all_epo, epo], dim='reg')

    return all_epo


def spectral_analysis(data, fs=1000.):
    values = [0, 1]
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for i, v in enumerate(values):
        _data = data.loc[{'reg': v}].mean('reg')
        # _data = data.loc[{'reg': v}]
        # calculate spectrogram
        f, t, spg = ss.spectrogram(_data, fs=fs, nperseg=50, noverlap=49)
        # spg = spg.mean(0)
        t += data.times.values[0]
        _spg = xr.DataArray(spg, coords=[f, t], dims=['freqs', 'times'])
        _spg = _spg.loc[{'freqs': slice(8, 50)}]
        _spg.plot.pcolormesh(x='times', y='freqs', ax=axs[i])
    plt.show()


def periodogram_analysis(data, fs=1000.):
    values = [0, 1]
    fig, ax = plt.subplots(1, 1)
    da = []
    for i, v in enumerate(values):
        # _data = data.loc[{'reg': v}].mean('reg')
        _data = data.loc[{'reg': v}]
        # calculate spectrogram
        f, pdg = ss.periodogram(_data, fs=fs, axis=-1, nfft=400,
                                scaling='spectrum', window='bartlett')
        # f, pdg = ss.periodogram(_data, fs=fs, axis=-1, scaling='spectrum',
        # window='flattop')
        # f, pdg = ss.welch(_data, fs=fs, window='flattop', nperseg=250,
        #                   noverlap=25, nfft=500, scaling='density', axis=-1)
        # f, pdg = ss.welch(_data, fs=fs, window='flattop', nperseg=1024,
        #                   scaling='spectrum', axis=-1)
        # spg = spg.mean(0)
        # pdg = np.sqrt(pdg)
        _pdg = xr.DataArray(pdg, coords=[_data.reg, f], dims=['reg', 'freqs'])
        _pdg = _pdg.loc[{'freqs': slice(5, 55)}].mean('reg')
        da.append(_pdg)
    da = xr.concat(da, dim='Reward value')
    print(da.shape)
    # for _d in da:
    #     plt.semilogy(da.freqs, np.sqrt(_d))
    da.plot.line(x='freqs', ax=ax, yscale='log')
    plt.show()


if __name__ == '__main__':
    monkeys = ['freddie', 'teddy']
    conditions = ['easy', 'hard']
    event = 'trig_off'
    # norm = 'fbline_relchange'
    file = '{0}_trig_off_epo.fif'
    bline = '{0}_cue_on_epo.fif'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    # sectors = ['motor striatum', 'limbic striatum']

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
        # log_bads = []
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
                # fid = fid[fid['neuron_type'] == 'TAN']

                # all_files = []
                # all_bline = []
                # all_conds = []
                # log_bads = []
                # bad_epo = []
                for fs in fid['file']:
                    fname = op.join(epo_dir, file.format(fs))
                    bname = op.join(epo_dir, bline.format(fs))
                    rname = op.join(regr_dir, '{0}.xlsx'.format(fs))
                    if op.exists(fname) and fs not in rej_files:
                    # if op.exists(fname) and fs in acc_files:
                        all_files.append(fname)
                        all_bline.append(bname)
                        all_conds.append(rname)

                        # fname_epo = op.join(epo_dir,
                        #                     '{0}_{1}_epo.fif'.format(fs,
                        #                                              event))
                        # lb = get_log_bad_epo(fname_epo)
                        # log_bads.append(lb)
                        be = get_ch_bad_epo(monkey, condition, fs,
                                            fname_info=rec_info)
                        bad_epo.append(be)
                        # plot_avg_tf([fname], [bname])

        data = prepare_data(all_files, bad_epo, all_conds, fbline=all_bline,
                            t_win=(.0, .8), t_bln=(-.6, -0.1))
        # data = prepare_data(all_files, bad_epo, all_conds, fbline=None,
        #                     t_win=(.3, .5))
        # spectral_analysis(data)
        periodogram_analysis(data)
