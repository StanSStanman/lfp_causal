import numpy as np
import mne
import pandas as pd
import xarray as xr
import os
import os.path as op
import matplotlib.pyplot as plt
from lfp_causal.tf_analysis import epochs_tf_analysis
from lfp_causal.aslt import parallel_aslt
from lfp_causal.IO import read_sector
from lfp_causal.compute_bad_epochs import get_ch_bad_epo


def compute_power_morlet(epoch, session, event, crop=None, freqs=None,
                         n_cycles=None):
    csv_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/' \
              'lfp_causal/{0}/{1}/t_events'

    # for epo, ses in zip(epochs, sessions):
    if isinstance(epoch, str):
        epoch = mne.read_epochs(epoch, preload=True)

    monkey = epoch.filename.split('/')[-4]
    condition = epoch.filename.split('/')[-3]
    csv_dir = csv_dir.format(monkey, condition)
    csv = pd.read_csv(op.join(csv_dir, '{0}.csv'.format(session)))

    if crop is not None:
        assert isinstance(crop, tuple), \
            AssertionError('frequencies should be expressed as a '
                           'tuple(tmin, tmax)')
        epoch.crop(crop[0], crop[1])

    if freqs is not None:
        assert isinstance(freqs, (tuple, np.ndarray)), \
            AssertionError('frequencies should be expressed as a '
                           'tuple(fmin, fmax), or array type')
        # epoch.filter(freqs[0], freqs[1])

    bads = get_ch_bad_epo(monkey, condition, session)
    _b = []
    for i, d in enumerate(epoch.drop_log):
        if d != ():
            _b.append(i)

    movements = csv['mov_on'].values
    movements = np.delete(movements, _b)
    if 'mov_on' not in epoch.filename:
        epoch = epoch[np.isfinite(movements)]

    epoch.drop(bads)

    d_tfr, _, _ = epochs_tf_analysis(epoch, t_win=crop, freqs=freqs,
                                     n_cycles=n_cycles, avg=False, show=False,
                                     baseline=(-2., -1.7))
    plt.close('all')
    f_range = '{0}_{1}'.format(np.round(d_tfr.freqs[0]).astype(int),
                               np.round(d_tfr.freqs[-1]).astype(int))

    d_tfr = xr.DataArray(d_tfr.data.squeeze(),
                         coords=[range(d_tfr.__len__()),
                                 d_tfr.freqs,
                                 d_tfr.times],
                         dims=['trials', 'freqs', 'times'],
                         name=session)

    p_dir = epoch.filename.replace(epoch.filename.split('/')[-1], '') \
        .replace('epo', 'pow/{0}'.format(session))
    p_fname = op.join(p_dir, '{0}_pow_{1}.nc'.format(event, f_range))
    if not op.exists(p_dir):
        os.makedirs(p_dir)
    d_tfr.to_netcdf(p_fname)

    return


def compute_power_superlets(epoch, session, event, freqs,
                            n_cycles=None, crop=None, cut=.15):
    csv_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/' \
              'lfp_causal/{0}/{1}/t_events'

    # for epo, ses in zip(epochs, sessions):
    if isinstance(epoch, str):
        epoch = mne.read_epochs(epoch, preload=True)

    monkey = epoch.filename.split('/')[-4]
    condition = epoch.filename.split('/')[-3]
    csv_dir = csv_dir.format(monkey, condition)
    csv = pd.read_csv(op.join(csv_dir, '{0}.csv'.format(session)))

    if crop is not None:
        assert isinstance(crop, tuple), \
            AssertionError('frequencies should be expressed as a '
                           'tuple(tmin, tmax)')
        epoch.crop(crop[0], crop[1])

    bads = get_ch_bad_epo(monkey, condition, session)
    _b = []
    for i, d in enumerate(epoch.drop_log):
        if d != ():
            _b.append(i)

    movements = csv['mov_on'].values
    movements = np.delete(movements, _b)
    if 'mov_on' not in epoch.filename:
        epoch = epoch[np.isfinite(movements)]

    epoch.drop(bads)
    epoch.drop_channels(epoch.info['bads'])

    if n_cycles is None:
        n_cycles = freqs / 2.
    elif isinstance(n_cycles, (int, float)):
        n_cycles = np.full_like(freqs, n_cycles)
    times = epoch.times

    print('Computing power for session {0}.....'.format(session))
    sl_pow = parallel_aslt(epoch.get_data().squeeze(), epoch.info['sfreq'],
                           freqs, n_cycles, times,
                           order=[1, 30], mult=True, n_jobs=-1)
    sl_pow = sl_pow.rename(session)
    tmin, tmax = sl_pow.times[0] + cut, sl_pow.times[-1] - cut
    sl_pow = sl_pow.loc[dict(times=slice(tmin, tmax))]
    print('Done.\n')

    f_range = '{0}_{1}'.format(freqs[0].astype(int), freqs[-1].astype(int))
    p_dir = epoch.filename.replace(epoch.filename.split('/')[-1], '') \
        .replace('epo', 'pow/{0}'.format(session))
    p_fname = op.join(p_dir, '{0}_pow_{1}_sl.nc'.format(event, f_range))
    if not op.exists(p_dir):
        os.makedirs(p_dir)
    sl_pow.to_netcdf(p_fname)

    return


def compute_power_multitaper(epoch, session, event, crop=None):
    csv_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/' \
              'lfp_causal/{0}/{1}/t_events'
    freqs = [11., 25., 55., 95.]
    n_cycles = [6., 7., 16., 28.]
    t_bandwidth = [4., 6., 8., 14.]
    # freqs = [30.] #[25.]
    # n_cycles = [freqs[0] / 3.]
    # t_bandwidth = [5.]

    # for epo, ses in zip(epochs, sessions):
    if isinstance(epoch, str):
        epoch = mne.read_epochs(epoch, preload=True)

    monkey = epoch.filename.split('/')[-4]
    condition = epoch.filename.split('/')[-3]
    csv_dir = csv_dir.format(monkey, condition)
    csv = pd.read_csv(op.join(csv_dir, '{0}.csv'.format(session)))

    if crop is not None:
        assert isinstance(crop, tuple), \
            AssertionError('frequencies should be expressed as a '
                           'tuple(tmin, tmax)')
        epoch.crop(crop[0], crop[1])

    # if freqs is not None:
    #     assert isinstance(freqs, tuple), \
    #         AssertionError('frequencies should be expressed as a '
    #                        'tuple(fmin, fmax)')
    #     epoch.filter(freqs[0], freqs[1])

    bads = get_ch_bad_epo(monkey, condition, session)
    _b = []
    for i, d in enumerate(epoch.drop_log):
        if d != ():
            _b.append(i)

    movements = csv['mov_on'].values
    movements = np.delete(movements, _b)
    if 'mov_on' not in epoch.filename:
        epoch = epoch[np.isfinite(movements)]

    epoch.drop(bads)

    data = []
    for _f, _nc, _tb in zip(freqs, n_cycles, t_bandwidth):
        d_tfr = mne.time_frequency.tfr_multitaper(epoch, freqs=[_f],
                                                  n_cycles=[_nc],
                                                  time_bandwidth=_tb,
                                                  use_fft=True,
                                                  return_itc=False,
                                                  average=False, n_jobs=-1)
        data.append(d_tfr.data)
    data = np.concatenate(data, axis=1)

    # d_tfr, _, _ = epochs_tf_analysis(epoch, t_win=crop, freqs=freqs,
    #                                  avg=False, show=False,
    #                                  baseline=(-2., -1.7))
    # plt.close('all')
    f_range = '8_120'
    data = data.squeeze()

    # f_range = 'beta'
    # data = data.squeeze(axis=1)

    d_tfr = xr.DataArray(data,
                         coords=[range(d_tfr.__len__()),
                                 freqs,
                                 d_tfr.times],
                         dims=['trials', 'freqs', 'times'],
                         name=session)

    p_dir = epoch.filename.replace(epoch.filename.split('/')[-1], '') \
        .replace('epo', 'pow/{0}'.format(session))
    p_fname = op.join(p_dir, '{0}_pow_{1}_mt.nc'.format(event, f_range))
    if not op.exists(p_dir):
        os.makedirs(p_dir)
    d_tfr.to_netcdf(p_fname)
    print('Saved at', p_fname)

    return


def normalize_power(power, norm, bline=(-.2, 0.), file=None):

    if norm.startswith('fbline'):
        if file is None:
            raise ValueError('To use {0} as normalization, file must be a '
                             'baseline filename'.format(norm))
        b = xr.load_dataset(file)
        b = b.loc[dict(times=slice(bline[0], bline[1]))]
        b = np.array(b.to_array()).squeeze(axis=0)

    # Data are in the shape (trials, freqs, times)
    data = np.array(power.to_array()).squeeze(axis=0)
    name = list(power.keys())[0]
    trials = power.trials.values
    freqs = power.freqs.values
    times = power.times.values

    if norm == 'log':
        # data = np.log(data)
        np.log(data, out=data)

    elif norm == 'log10':
        # data = np.log10(data)
        np.log10(data, out=data)

    elif norm == 'relchange':
        m = data.mean(2, keepdims=True)
        # data = (data - m) / m
        np.divide(np.subtract(data, m, out=data), m, out=data)

    elif norm == 'db':
        data = 10 * np.log10(data / data.mean(2, keepdims=True))

    elif norm == 'zscore':
        data = ((data - data.mean(2, keepdims=True)) /
                data.std(2, keepdims=True))

    elif norm == 'bline':
        b = power.loc[dict(times=slice(bline[0], bline[1]))]
        b = np.array(b.to_array()).squeeze()
        data = data / b.mean(2, keepdims=True)

    elif norm == 'bline_log':
        b = power.loc[dict(times=slice(bline[0], bline[1]))]
        b = np.array(b.to_array()).squeeze()
        data = np.log(data) - np.log(b.mean(2, keepdims=True))

    elif norm == 'bline_zs':
        b = power.loc[dict(times=slice(bline[0], bline[1]))]
        b = np.array(b.to_array()).squeeze()
        data = (data - b.mean(2, keepdims=True)) / b.std(2, keepdims=True)

    elif norm == 'bline_tt':
        b = power.loc[dict(times=slice(bline[0], bline[1]))]
        b = np.array(b.to_array()).squeeze()
        data = data / b.mean(2, keepdims=True).mean(0, keepdims=True)

    elif norm == 'bline_tt_log':
        b = power.loc[dict(times=slice(bline[0], bline[1]))]
        b = np.array(b.to_array()).squeeze()
        data = np.log(data) - \
            np.log(b.mean(2, keepdims=True).mean(0, keepdims=True))

    elif norm == 'bline_tt_zs':
        b = power.loc[dict(times=slice(bline[0], bline[1]))]
        b = np.array(b.to_array()).squeeze()
        data = (data - b.mean(2, keepdims=True).mean(0, keepdims=True)) / \
            b.std(2, keepdims=True).mean(0, keepdims=True)

    elif norm == 'fbline':
        # b = xr.load_dataset(file)
        # b = b.loc[dict(times=slice(bline[0], bline[1]))]
        # b = np.array(b.to_array()).squeeze()
        data = (data - b.mean(2, keepdims=True)) / b.mean(2, keepdims=True)

    elif norm == 'fbline_zs':
        # b = xr.load_dataset(file)
        # b = b.loc[dict(times=slice(bline[0], bline[1]))]
        # b = np.array(b.to_array()).squeeze()
        data = (data - b.mean(2, keepdims=True)) / b.std(2, keepdims=True)

    elif norm == 'fbline_tt':
        # b = xr.load_dataset(file)
        # b = b.loc[dict(times=slice(bline[0], bline[1]))]
        # b = np.array(b.to_array()).squeeze()
        data = data / b.mean(2, keepdims=True).mean(0, keepdims=True)

    elif norm == 'fbline_tt_zs':
        # b = xr.load_dataset(file)
        # b = b.loc[dict(times=slice(bline[0], bline[1]))]
        # b = np.array(b.to_array()).squeeze()
        data = (data - b.mean(2, keepdims=True).mean(0, keepdims=True)) / \
            b.std(2, keepdims=True).mean(0, keepdims=True)

    elif norm == 'fbline_relchange':
        data = (data - b.mean(2, keepdims=True)) / b.mean(2, keepdims=True)

    power = xr.DataArray(data, coords=[trials, freqs, times],
                         dims=['trials', 'freqs', 'times'],
                         name=name).to_dataset()
    return power


if __name__ == '__main__':

    monkeys = ['freddie', 'teddy']
    conditions = ['easy', 'hard']
    event = 'cue_on'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    # sectors = ['motor striatum']
    # sectors = ['associative striatum']
    # sectors = ['limbic striatum']
    freqs = np.geomspace(8, 120, num=80)
    n_cycles = freqs / 4 #np.geomspace(3, 12, num=80)

    for monkey in monkeys:
        for condition in conditions:

            rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
                       '{0}/{1}/files_info.xlsx'.format(monkey, condition)
            epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
                      'lfp_causal/{0}/{1}/epo'.format(monkey, condition)
            eve_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
                      'lfp_causal/{0}/{1}/eve'.format(monkey, condition)

            # rej_ses = ['0831', '0837', '0981', '1043']
            rej_ses = []

            epo_fname = []
            ses_n = []
            for sect in sectors:
                fid = read_sector(rec_info, sect)

                for file in fid['file']:
                    # file = '0610'
                    if file not in rej_ses:
                        # fname_epo = op.join(epo_dir,
                        #                     '{0}_{1}_epo.fif'.format(file, event))
                        fname_epo_t = op.join(epo_dir,
                                            '{0}_{1}_epo.fif'.format(file, 'trig_off'))
                        fname_epo_c = op.join(epo_dir,
                                            '{0}_{1}_epo.fif'.format(file, 'cue_on'))
                        if op.exists(fname_epo_t):
                            # # TRIGGER OFFSET
                            # compute_power_morlet(fname_epo_t, file, 'trig_off',
                            #                      freqs=freqs,
                            #                      n_cycles=n_cycles,
                            #                      crop=(-1.8, 1.45))
                            # # CUE ONSET
                            # compute_power_morlet(fname_epo_c, file, 'cue_on',
                            #                      freqs=freqs,
                            #                      n_cycles=n_cycles,
                            #                      crop=(-.75, -.15))

                            ## TRIGGER OFFSET
                            # compute_power_superlets(fname_epo, file, event,
                            #                         freqs=np.linspace(8, 120, 80),
                            #                         n_cycles=None,
                            #                         crop=(-1.8, 1.45))
                            ## CUE ONSET
                            # compute_power_superlets(fname_epo, file, event,
                            #                         freqs=np.linspace(8, 120, 80),
                            #                         n_cycles=None,
                            #                         crop=(-.75, .15))
                            ## TRIGGER ONSET
                            # compute_power_superlets(fname_epo, file, event,
                            #                         freqs=np.linspace(8, 120, 80),
                            #                         n_cycles=None,
                            #                         crop=(-1.5, -1.5))

                            # TRIGGER OFFSET
                            compute_power_multitaper(fname_epo_t, file,
                                                     'trig_off',
                                                     crop=(-1.8, 1.45))
                            # CUE ONSET
                            compute_power_multitaper(fname_epo_c, file,
                                                     'cue_on',
                                                     crop=(-.75, -.15))
                            ## TRIGGER ONSET
                            # compute_power_multitaper(fname_epo, file, event,
                            #                         crop=(-1.7, 1.85))
