import os
import os.path as op
import xarray as xr
import numpy as np
import pandas as pd
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
from itertools import product
from research.get_dirs import get_dirs
from lfp_causal.IO import read_session
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo
from lfp_causal.compute_power import normalize_power
# os.system("taskset -p 0xff %d" % os.getpid())
from lfp_causal.profiling import (RepeatedTimer, memory_usage, cpu_usage)
import time
import json


def prepare_data(powers, regressors, l_bad, e_bad, conditions, reg_name,
                 reg_val, times=None, freqs=None, avg_freq=False,
                 norm=None, bline=None, fbl=None):

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

    all_pow = []
    for p in powers:
    # for p, r, lb, eb in zip(powers, regresors, l_bad, e_bad):
        print('Opening', p)
        pow = xr.open_dataset(p)

        if fbl is not None:
            _fbl = p.split('/')[:-1] + [fbl]
            _fbl = '/' + op.join(*_fbl)

        if norm is not None:
            pow = normalize_power(pow, norm, bline, _fbl)

        if isinstance(times, (tuple, list, np.ndarray)):
            pow = pow.loc[dict(times=slice(tmin, tmax))]
        all_times = pow.times.values

        if isinstance(freqs, tuple):
            pow = pow.loc[dict(freqs=slice(fmin, fmax))]
        all_freqs = pow.freqs.values

        pow = pow.to_array().values.transpose(1, 0, 2, 3)
        # pow = pow[nans, :, :, :]

        if avg_freq:
            pow = pow.mean(2)

        all_pow.append(pow)

    all_reg = {}
    for rn in reg_name:
        # for rv in reg_val:
        _reg = []
        for r, lb, eb, cn, _idx in zip(regressors, l_bad, e_bad,
                                       conditions, range(len(regressors))):

            xls = pd.read_excel(r, index_col=0)
            reg = xls[rn].values

            if len(lb) != 0:
                reg = np.delete(reg, lb)
            if len(eb) != 0:
                reg = np.delete(reg, eb)

            all_pow[_idx] = np.delete(all_pow[_idx], np.where(reg != reg_val),
                                      axis=0)
            reg = np.delete(reg, np.where(reg != reg_val))
            if cn == 'easy':
                reg = np.full_like(reg, 0)
            elif cn == 'hard':
                reg = np.full_like(reg, 1)

            _reg.append(reg)

        all_reg['{0}_{1}'.format(rn, reg_val)] = _reg

    return all_pow, all_reg, all_times, all_freqs


def compute_stats_meso(fname_pow, fname_reg, rois, log_bads, bad_epo,
                       conditions, regressor, reg_vals, mi_type, inference,
                       times=None, freqs=None, avg_freq=True, norm=None):

    power, regr, \
        times, freqs = prepare_data(fname_pow, fname_reg, log_bads,
                                    bad_epo, conditions=conditions,
                                    reg_name=regressor, reg_val=reg_vals,
                                    times=times, freqs=freqs,
                                    avg_freq=avg_freq,
                                    norm=norm, bline=(-.55, -0.05),
                                    fbl='cue_on_pow_8_120_mt.nc')

    mi_results = {}
    pv_results = {}
    conj_ss_results = {}
    conj_results = {}
    for _r, _mt, _inf in zip(regr, mi_type, inference):
        if _mt == 'cc':
            regr[_r] = [r.astype('float32') for r in regr[_r]]
        elif _mt == 'cd':
            regr[_r] = [r.astype('int32') for r in regr[_r]]
        elif _mt == 'ccd':
            regr[_r] = [r.astype('float32') for r in regr[_r]]
            cond[_r] = [c.astype('int32') for c in cond[_r]]

        ds_ephy = DatasetEphy(x=power.copy(), y=regr[_r], roi=rois,
                              z=None, times=times)

        wf = WfMi(mi_type=_mt, inference=_inf, kernel=np.hanning(20))
        mi, pval = wf.fit(ds_ephy, n_perm=1000, n_jobs=-1)
        mi['times'] = times
        pval['times'] = times

        if _inf == 'rfx':
            conj_ss, conj = wf.conjunction_analysis(ds_ephy)

        if not avg_freq:
            mi.assign_coords({'freqs': freqs})
            pval.assign_coords({'freqs': freqs})

        mi_results[_r] = mi
        pv_results[_r] = pval
        if _inf == 'rfx':
            conj_ss_results[_r] = conj_ss
            conj_results[_r] = conj

    ds_mi = xr.Dataset(mi_results)
    ds_pv = xr.Dataset(pv_results)
    if len(conj_ss_results) == len(conj_results) == 0:
        return ds_mi, ds_pv
    else:
        ds_conj_ss = xr.Dataset(conj_ss_results)
        ds_conj = xr.Dataset(conj_results)
        return ds_mi, ds_pv, ds_conj_ss, ds_conj


if __name__ == '__main__':
    from lfp_causal import MCH, PRJ
    dirs = get_dirs(MCH, PRJ)

    monkeys = ['freddie']
    conditions = ['easy', 'hard']
    event = 'trig_on'
    norm = 'fbline_relchange'
    n_power = '{0}_pow_8_120_mt.nc'.format(event)
    times = [(-1.5, 1.3)]
    # freqs = [(5, 120)]
    # freqs = [(8, 15), (15, 30), (25, 45), (40, 70), (60, 120)]
    freqs = [(8, 12), (15, 35), (40, 65), (70, 120)]
    avg_frq = True

    regressors = ['Correct', 'Reward',
                  'is_R|C', 'is_nR|C', 'is_R|nC', 'is_nR|nC',
                  'RnR|C', 'RnR|nC',
                  '#R', '#nR', '#R|C', '#nR|C', '#R|nC', '#nR|nC',
                  'learn_5t', 'learn_2t', 'early_late_cons',
                  'P(R|C)', 'P(R|nC)', 'P(R|Cho)', 'P(R|A)',
                  'dP', 'log_dP', 'delta_dP',
                  'surprise', 'surprise_bayes', 'rpe',
                  'q_pcorr', 'q_pincorr', 'q_dP',
                  'q_entropy', 'q_rpe', 'q_absrpe',
                  'q_shann_surp', 'q_bayes_surp']

    # conditionals = [None, None,
    #                 None, None, None, None,
    #                 None, None,
    #                 None, None, None, None, None, None,
    #                 None, None, None,
    #                 None, None, None, None,
    #                 None, None, None,
    #                 None, None, None,
    #                 None, None, None,
    #                 None, None, None,
    #                 None, None]
    # conditionals = ['Condition' for r in regressors]

    mi_type = ['cd', 'cd',
               'cd', 'cd', 'cd', 'cd',
               'cd', 'cd',
               'cc', 'cc', 'cc', 'cc', 'cc', 'cc',
               'cd', 'cd', 'cd',
               'cc', 'cc', 'cc', 'cc',
               'cc', 'cc', 'cc',
               'cc', 'cc', 'cc',
               'cc', 'cc', 'cc',
               'cc', 'cc', 'cc',
               'cc', 'cc']
    # mi_type = ['ccd' for r in regressors]

    regressors = ['Reward']
    reg_vals = 0
    conditionals = [None]
    mi_type = ['cd']

    inference = ['ffx' for r in regressors]

    fn_pow_list = []
    fn_reg_list = []
    rois = []
    log_bads = []
    bad_epo = []
    conds = []

    rej_files = ['1204', '1217', '1231', '0944', # Bad sessions
                 '0845', '0847', '0939', '0946', '0963', '1036', '1231',
                 '1233', '1234', '1514', '1699',
                 '0940', '0944', '0964', '0967', '0969', '0970', '0971',
                 '0977', '0985', '1280']

    for monkey in monkeys:
        for condition in conditions:
            epo_dir = dirs['epo'].format(monkey, condition)
            power_dir = dirs['pow'].format(monkey, condition)
            regr_dir = dirs['reg'].format(monkey, condition)
            fname_info = op.join(dirs['ep_cnds'].format(monkey, condition),
                                 'files_info.xlsx')
            for d in os.listdir(power_dir):
                if d in rej_files:
                    continue
                if op.isdir(op.join(power_dir, d)):
                    fname_power = op.join(power_dir, d, n_power)
                    fname_regr = op.join(regr_dir, '{0}.xlsx'.format(d))
                    fname_epo = op.join(epo_dir,
                                        '{0}_{1}_epo.fif'.format(d, event))

                    fn_pow_list.append(fname_power)
                    fn_reg_list.append(fname_regr)
                    rois.append(read_session(fname_info, d)['sector'].values)

                    lb = get_log_bad_epo(fname_epo)
                    log_bads.append(lb)

                    be = get_ch_bad_epo(monkey, condition, d,
                                        fname_info=fname_info)
                    bad_epo.append(be)

                    conds.append(condition)

    mi_results = {}
    pv_results = {}
    for t, f in product(times, freqs):
        ds_mi, ds_pv = compute_stats_meso(fn_pow_list, fn_reg_list, rois,
                                          log_bads, bad_epo, conds,
                                          regressors, reg_vals,
                                          mi_type, inference,
                                          t, f, avg_frq, norm)

        mk = 'freddie'
        cd = '2cond_nrd'

        if avg_frq:
            save_dir = op.join(dirs['st_prj'], mk, cd, event, norm,
                               '{0}_{1}_mt'.format(f[0], f[1]))

        elif not avg_frq:
            save_dir = op.join(dirs['st_prj'], mk, cd, event, norm,
                               '{0}_{1}_tf_mt'.format(f[0], f[1]))

        os.makedirs(save_dir, exist_ok=True)

        ds_mi.to_netcdf(op.join(save_dir,
                                'mi_results.nc'.format(f[0], f[1])))
        ds_pv.to_netcdf(op.join(save_dir,
                                'pv_results.nc'.format(f[0], f[1])))
