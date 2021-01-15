import os
import os.path as op
import numpy as np
import xarray as xr
import pandas as pd
from scipy.signal import resample as spr
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
from lfp_causal.IO import read_session
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo

def prepare_data(powers, regresors, l_bad, e_bad, reg_name, cond=None,
                 times=None, freqs=None, avg_freq=False,
                 t_rsmpl=None, f_rsmpl=None):

    if avg_freq is True and f_rsmpl is not None:
        print('The mean values between frequecies will be taken, '
              'f_rsmpl will be considered automatically as None')
        f_rsmpl = None

    if times is not None:
        if isinstance(times, tuple):
            tmin, tmax = times
        elif isinstance(times, (list, np.ndarray)):
            tmin, tmax = times[0], times[-1]
        else:
            raise ValueError('times must be NoneType '
                             'tuple of values (tmin, tmax),'
                             'list or numpy array')

    if freqs is not None:
        if isinstance(freqs, tuple):
            fmin, fmax = freqs
        else:
            raise ValueError('freqs must be NoneType '
                             'or tuple of values (fmin, fmax)')

    all_pow, all_reg, all_con = [], [], None
    for p, r, lb, eb in zip(powers, regresors, l_bad, e_bad):
        print('Opening', p)
        pow = xr.open_dataset(p)

        xls = pd.read_excel(r, index_col=0)
        reg = xls[reg_name].values

        if len(lb) != 0:
            reg = np.delete(reg, lb)
        if len(eb) != 0:
            reg = np.delete(reg, eb)

        if np.sum(np.isfinite(reg)) < 2 or np.any(np.isinf(reg)):
            print('Skipped', p)
            continue

        nans, _nans = range(len(reg)), np.any(np.isnan(reg))
        if _nans:
            nans = np.isfinite(reg)

        if isinstance(times, (tuple, list, np.ndarray)):
            pow = pow.loc[dict(times=slice(tmin, tmax))]
        all_times = pow.times.values

        if isinstance(freqs, tuple):
            pow = pow.loc[dict(freqs=slice(fmin, fmax))]
        all_freqs = pow.freqs.values

        pow = pow.to_array().values.transpose(1, 0, 2, 3)
        pow = pow[nans, :, :, :]

        if avg_freq:
            pow = pow.mean(2)
        else:
            if f_rsmpl is not None:
                assert pow.shape[2] > f_rsmpl, \
                    ValueError('The number of resampled frequencies should be '
                               'lower than the number of actual frequencies')
                spacing = int(round(pow.shape[2] / f_rsmpl))
                pow = pow[:, :, ::spacing, :]
                all_freqs = all_freqs[::spacing]

        if t_rsmpl is not None:
            pow, all_times = spr(x=pow, num=t_rsmpl, t=all_times, axis=-1)

        all_pow.append(pow)

        # xls = pd.read_excel(r, index_col=0)
        # reg = xls[reg_name].values

        # reg = np.delete(reg, nans, axis=0)
        reg = reg[nans]
        all_reg.append(reg)

        assert pow.shape[0] == reg.shape[0]
        # print(pow.shape[0], reg.shape[0])

        if cond is not None:
            if all_con is None:
                all_con = []

            if isinstance(cond, str):
                con = xls[cond].values
            else:
                raise ValueError('cond must be NoneType or a str with the '
                                 'name of the variable used to condition '
                                 'your analysis')
            if len(lb) != 0:
                con = np.delete(con, lb)
            if len(eb) != 0:
                con = np.delete(con, eb)
            # con = np.delete(con, nans, axis=0)
            con = con[nans]
            all_con.append(con)

            assert pow.shape[0] == con.shape[0]

    # all_pow = np.concatenate(all_pow, axis=0)
    # all_reg = np.hstack(tuple(all_reg))
    # if all_con is not None:
    #     all_con = np.hstack(tuple(all_con))

    return all_pow, all_reg, all_con, all_times, all_freqs


if __name__ == '__main__':

    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_off'
    n_power = '{0}_pow_5_120.nc'.format(event)
    t_res = 0.001
    times = (-1., 1.3)
    freqs = (5, 120)

    # epo_dir = '/scratch/rbasanisi/data/db_lfp/' \
    #           'lfp_causal/{0}/{1}/epo'.format(monkey, condition)
    # power_dir = '/scratch/rbasanisi/data/db_lfp/lfp_causal/' \
    #             '{0}/{1}/pow'.format(monkey, condition)
    # regr_dir = '/scratch/rbasanisi/data/db_behaviour/lfp_causal/' \
    #            '{0}/{1}/regressors'.format(monkey, condition)
    # fname_info = '/scratch/rbasanisi/data/db_lfp/lfp_causal/' \
    #              '{0}/{1}/files_info.xlsx'.format(monkey, condition)

    epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)
    power_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
                '{0}/{1}/pow'.format(monkey, condition)
    regr_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/lfp_causal/' \
               '{0}/{1}/regressors'.format(monkey, condition)
    fname_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
                 '{0}/{1}/recording_info.xlsx'.format(monkey, condition)

    regressors = ['learn_5t', 'RnR|nC', 'q_dP', 'q_shann_surp']

    # regressors = ['Correct', 'Reward',
    #               'is_R|C', 'is_nR|C', 'is_R|nC', 'is_nR|nC',
    #               'RnR|C', 'RnR|nC',
    #               '#R', '#nR', '#R|C', '#nR|C', '#R|nC', '#nR|nC',
    #               'learn_5t', 'learn_2t', 'early_late_cons',
    #               'P(R|C)', 'P(R|nC)', 'P(R|Cho)', 'P(R|A)',
    #               'dP', 'log_dP', 'delta_dP',
    #               'surprise', 'surprise_bayes', 'rpe']

    fn_pow_list = []
    fn_reg_list = []
    rois = []
    log_bads = []
    bad_epo = []
    files = ['0814', '0822', '1043', '1191']
    for d in files:
    # for d in os.listdir(power_dir):
        if op.isdir(op.join(power_dir, d)):
            fname_power = op.join(power_dir, d, n_power)
            fname_regr = op.join(regr_dir, '{0}.xlsx'.format(d))
            fname_epo = op.join(epo_dir, '{0}_{1}_epo.fif'.format(d, event))

            fn_pow_list.append(fname_power)
            fn_reg_list.append(fname_regr)
            rois.append(read_session(fname_info, d)['sector'].values)

            lb = get_log_bad_epo(fname_epo)
            log_bads.append(lb)

            be = get_ch_bad_epo(monkey, condition, d)
            bad_epo.append(be)

    power, regr, cond, tpoints, vfreqs = prepare_data(fn_pow_list, fn_reg_list,
                                                      log_bads, bad_epo,
                                                      'Reward', cond=None,
                                                      times=times, freqs=freqs,
                                                      avg_freq=False,
                                                      t_rsmpl=1000, f_rsmpl=50)

    # t_points = np.arange(times[0], times[1] + t_res, t_res)
    ds_ephy = DatasetEphy(x=power, y=regr, roi=rois, z=cond, times=tpoints)

    wf = WfMi(mi_type='cd', inference='ffx')
    mi, pval = wf.fit(ds_ephy, n_perm=1000)

    import matplotlib.pyplot as plt
    for r in range(mi.shape[1]):
        plt.plot(mi[:, r])
        sig = mi[:, r]
        sig[pval[:, r] > 0.05] = np.nan
        plt.plot(sig, color='r', linewidth=2.5)
    plt.show()
