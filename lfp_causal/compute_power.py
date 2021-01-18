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

    p_dir = epoch.filename.replace(epoch.filename.split('/')[-1], '')\
        .replace('epo', 'pow/{0}'.format(session))
    p_fname = op.join(p_dir, '{0}_pow_{1}.nc'.format(event, f_range))
    if not op.exists(p_dir):
        os.makedirs(p_dir)
    d_tfr.to_netcdf(p_fname)

    return


if __name__ == '__main__':

    monkey = 'freddie'
    condition = 'hard'
    event = 'trig_off'
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
                                  freqs=(5, 120), crop=(-2, 1.5))
