import numpy as np
import mne
import pandas as pd
import xarray as xr
import os
import os.path as op
import matplotlib.pyplot as plt
from lfp_causal.tf_analysis import epochs_tf_analysis
from lfp_causal.IO import read_sector
from lfp_causal.compute_bad_epochs import get_ch_bad_epo


def compute_power(epoch, session, event, crop=None, freqs=None):
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
        assert isinstance(freqs, tuple), \
            AssertionError('frequencies should be expressed as a '
                           'tuple(fmin, fmax)')
        epoch.filter(freqs[0], freqs[1])

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
                                     avg=False, show=False,
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


def normalize_power(power, norm, bline=(-.2, 0.), file=None):

    if norm.startswith('fbline') and file is None:
        raise ValueError('To use {0} as normalization, file must be a '
                         'baseline filename'.format(norm))

    # Data are in the shape (trials, freqs, times)
    data = np.array(power.to_array()).squeeze()
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
        b = xr.load_dataset(file)
        b = b.loc[dict(times=slice(bline[0], bline[1]))]
        b = np.array(b.to_array()).squeeze()
        data = data / b.mean(2, keepdims=True)

    elif norm == 'fbline_zs':
        b = xr.load_dataset(file)
        b = b.loc[dict(times=slice(bline[0], bline[1]))]
        b = np.array(b.to_array()).squeeze()
        data = (data - b.mean(2, keepdims=True)) / b.std(2, keepdims=True)

    elif norm == 'fbline_tt':
        b = xr.load_dataset(file)
        b = b.loc[dict(times=slice(bline[0], bline[1]))]
        b = np.array(b.to_array()).squeeze()
        data = data / b.mean(2, keepdims=True).mean(0, keepdims=True)

    elif norm == 'fbline_tt_zs':
        b = xr.load_dataset(file)
        b = b.loc[dict(times=slice(bline[0], bline[1]))]
        b = np.array(b.to_array()).squeeze()
        data = (data - b.mean(2, keepdims=True).mean(0, keepdims=True)) / \
            b.std(2, keepdims=True).mean(0, keepdims=True)

    power = xr.DataArray(data, coords=[trials, freqs, times],
                         dims=['trials', 'freqs', 'times'],
                         name=name).to_dataset()
    return power


if __name__ == '__main__':

    monkey = 'freddie'
    condition = 'hard'
    event = 'cue_on'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    # sectors = ['motor striatum']
    # sectors = ['associative striatum']
    # sectors = ['limbic striatum']

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
            # file = '1280'
            if file not in rej_ses:
                fname_epo = op.join(epo_dir,
                                    '{0}_{1}_epo.fif'.format(file, event))
                if op.exists(fname_epo):
                    compute_power(fname_epo, file, event,
                                  freqs=(5, 120), crop=(-.75, .15))
