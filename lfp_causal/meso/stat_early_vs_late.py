import numpy as np
import xarray as xr
import os
import os.path as op
import pandas as pd
from itertools import product
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
from lfp_causal.compute_power import normalize_power
from lfp_causal.IO import read_session
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo
from research.get_dirs import get_dirs

def prepare_data_evl(powers, times=None, freqs=None, avg_freq=False,
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
    all_reg = []
    for p in powers:
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
        # print(pow.shape[0])
        # pow = pow[nans, :, :, :]

        early_pow = pow[:5, :, :, :]
        late_pow = pow[-5:, :, :, :]
        pow = np.concatenate((early_pow, late_pow), axis=0)

        if avg_freq:
            pow = pow.mean(2)

        all_pow.append(pow)

        reg = np.zeros(10, dtype=int)
        reg[-5:] = 1
        all_reg.append(reg)
    # all_reg = {'evl': all_reg}

    return all_pow, all_reg, all_times, all_freqs


def compute_stat_evl(fname_pow, rois, times=None, freqs=None, avg_freq=True,
                     norm=None):
    power, regr, times, freqs = prepare_data_evl(fname_pow, times=times,
                                                 freqs=freqs,
                                                 avg_freq=avg_freq,
                                                 norm=norm,
                                                 bline=(-.55, -0.05),
                                                 fbl='cue_on_pow_8_120_mt.nc')

    mi_results = {}
    pv_results = {}

    ds_ephy = DatasetEphy(x=power.copy(), y=regr.copy(), roi=rois,
                          z=None, times=times)

    wf = WfMi(mi_type='cd', inference='ffx', kernel=np.hanning(20))
    mi, pval = wf.fit(ds_ephy, n_perm=1000, n_jobs=-1)

    mi['times'] = times
    pval['times'] = times

    if not avg_freq:
        mi.assign_coords({'freqs': freqs})
        pval.assign_coords({'freqs': freqs})

    mi_results['evl'] = mi
    pv_results['evl'] = pval

    ds_mi = xr.Dataset(mi_results)
    ds_pv = xr.Dataset(pv_results)

    return ds_mi, ds_pv


def find_val_change(s_arr, t_arr=None, return_cutpoint=False):
    if t_arr is None:
        t_arr = s_arr.copy()

    assert len(s_arr) == len(t_arr), 'source and target array should ' \
                                     'have the same dimensions'
    cut_arr = []
    cut_p = []
    cn = s_arr[0]
    _i = 0
    n = len(s_arr)
    for i in range(n):
        nn = s_arr[i]
        if nn != cn:
            cn = nn
            cut_arr.append(t_arr[_i:i])
            cut_p.append(i)
            _i = i
        elif nn == cn:
            continue
    cut_arr.append(t_arr[_i:i + 1])
    cut_p.append(i + 1)
    if return_cutpoint:
        return cut_arr, cut_p
    elif not return_cutpoint:
        return cut_arr


def prepare_data_exp(powers, regressors, l_bad, e_bad, condition, reg_name,
                     rew_val, times=None, freqs=None, avg_freq=False,
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

    all_pow, all_reg, all_con = [], [], []
    for p, r, lb, eb in zip(powers, regressors, l_bad, e_bad):
        # for p, r, lb, eb in zip(powers, regresors, l_bad, e_bad):
        print('Opening', p)
        pow = xr.open_dataset(p)

        xls = pd.read_excel(r, index_col=0)
        reward = xls['Reward'].values
        reg = xls[reg_name].values

        if len(lb) != 0:
            reg = np.delete(reg, lb)
            reward = np.delete(reward, lb)
        if len(eb) != 0:
            reg = np.delete(reg, eb)
            reward = np.delete(reward, eb)

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

        reg = np.delete(reg, np.where(reward != rew_val)[0], axis=0)
        pow = np.delete(pow, np.where(reward != rew_val)[0], axis=0)

        ##########################
        # if pow.shape[0] > 25:
        #     pow = pow[:25, :, :]
        # if reg.shape[0] > 25:
        #     reg = reg[:25]
        ##########################

        if condition is not None:
            con = xls[condition].values
            if len(lb) != 0:
                con = np.delete(con, lb)
            if len(eb) != 0:
                con = np.delete(con, eb)
            con = np.delete(con, np.where(reward == rew_val)[0], axis=0)

        all_pow.append(pow)
        all_reg.append(reg)
        if condition is not None:
            all_con.append(con)

    if condition is None:
        return all_pow, all_reg, None, all_times, all_freqs
    elif condition is not None:
        return all_pow, all_reg, all_con, all_times, all_freqs


def compute_stat_exp(fname_pow, fname_reg, rois, log_bads, bad_epo,
                     times=None, freqs=None, avg_freq=True, norm=None):

    power, regr, cond, \
    times, freqs = prepare_data_exp(fname_pow, fname_reg,
                                    log_bads, bad_epo,
                                    condition=None,
                                    reg_name='Correct',
                                    rew_val=1,
                                    times=times,
                                    freqs=freqs,
                                    avg_freq=avg_freq,
                                    norm=norm,
                                    bline=(-.55, -0.05),
                                    fbl='cue_on_pow_8_120_mt.nc')

    mi_results = {}
    pv_results = {}

    ds_ephy = DatasetEphy(x=power.copy(), y=regr.copy(), roi=rois,
                          z=cond, times=times)

    wf = WfMi(mi_type='cd', inference='ffx', kernel=np.hanning(20))
    mi, pval = wf.fit(ds_ephy, n_perm=1000, n_jobs=-1)

    mi['times'] = times
    pval['times'] = times

    if not avg_freq:
        mi.assign_coords({'freqs': freqs})
        pval.assign_coords({'freqs': freqs})

    mi_results['corr_r1'] = mi
    pv_results['corr_r1'] = pval

    # when reg_name='learn_5t'
    # mi_results['expexp'] = mi
    # pv_results['expexp'] = pval

    # when reg_name='P(R|A)'
    # mi_results['pra_rew'] = mi
    # pv_results['pra_rew'] = pval

    ds_mi = xr.Dataset(mi_results)
    ds_pv = xr.Dataset(pv_results)

    return ds_mi, ds_pv


if __name__ == '__main__':
    import time
    # time.sleep(60*15)
    from lfp_causal import MCH, PRJ
    dirs = get_dirs(MCH, PRJ)

    monkeys = ['teddy']
    conditions = ['easy', 'hard']
    event = 'trig_off'
    norm = 'fbline_relchange'
    n_power = '{0}_pow_beta_mt.nc'.format(event)
    times = [(-1.5, 1.3)]
    # times = [(-1.5, 1.7)]
    # freqs = [(5, 120)]
    # freqs = [(8, 15), (15, 30), (25, 45), (40, 70), (60, 120)]
    freqs = [(8, 12), (15, 35), (40, 65), (70, 120)]
    avg_frq = True
    t_resample = None #1400
    f_resample = None #80
    overwrite = False

    # conditionals = [None]
    # mi_type = ['cd']
    # inference = ['ffx']
    #
    # fn_pow_list = []
    # fn_reg_list = []
    # rois = []
    # log_bads = []
    # bad_epo = []

    for condition in conditions: # Try to compute multiple conditions

        conditionals = [None]
        mi_type = ['cd']
        inference = ['ffx']

        fn_pow_list = []
        fn_reg_list = []
        rois = []
        log_bads = []
        bad_epo = []

        rej_files = []
        rej_files += ['1204', '1217', '1231', '0944', # Bad sessions
                      '0845', '0847', '0939', '0946', '0963', '1036', '1231',
                      '1233', '1234', '1514', '1699',

                      '0940', '0944', '0964', '0967', '0969', '0970', '0971',
                      '0977', '0985', '1037', '1280']
        rej_files += ['0210', '0219', '0221', '0225', '0226', '0227', '0230',
                      '0252', '0268', '0276', '0277', '0279', '0281', '0282',
                      '0283', '0285', '0288', '0290', '0323', '0362', '0365',
                      '0393', '0415', '0447', '0449', '0450', '0456', '0541',
                      '0573', '0622', '0628', '0631', '0643', '0648', '0653',
                      '0660', '0688', '0689', '0690', '0692', '0697', '0706',
                      '0710', '0717', '0718', '0719', '0713', '0726', '0732',

                      '0220', '0223', '0271', '0273', '0275', '0278', '0280',
                      '0284', '0289', '0296', '0303', '0363', '0416', '0438',
                      '0448', '0521', '0618', '0656', '0691', '0693', '0698',
                      '0705', '0707', '0711', '0712', '0716', '0720', '0731']

        rej_files += ['0900', '1512', '1555', '1682',
                      '0291', '0368', '0743']

        for monkey in monkeys:

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
                    fname_epo = op.join(epo_dir, '{0}_{1}_epo.fif'.format(d,
                                                                          event))

                    fn_pow_list.append(fname_power)
                    fn_reg_list.append(fname_regr)
                    # rois.append(['unique'])
                    rois.append(read_session(fname_info, d)['sector'].values)

                    lb = get_log_bad_epo(fname_epo)
                    log_bads.append(lb)

                    be = get_ch_bad_epo(monkey, condition, d,
                                        fname_info=fname_info)
                    bad_epo.append(be)

            mi_results = {}
            pv_results = {}
            for t, f in product(times, freqs):
                # ds_mi, ds_pv = compute_stat_evl(fn_pow_list, rois, t, f,
                #                                 avg_frq, norm)
                ds_mi, ds_pv = compute_stat_exp(fn_pow_list, fn_reg_list, rois,
                                                log_bads, bad_epo,
                                                times=t, freqs=f,
                                                avg_freq=True, norm=norm)

                ##################
                # if len(monkeys) > 1:
                #     monkey = 'freted'

                # if 'easy' in condition:
                #     condition = 'easy_25'
                # elif 'hard' in condition:
                #     condition = 'hard_25'
                # elif 'cued' in condition:
                #     condition = 'cued_25'
                ##################
                # condition = 'cued_cond'

                if avg_frq:
                    save_dir = op.join(dirs['st_prj'], monkey, condition, event,
                                       norm, '{0}_{1}_mt'.format(f[0], f[1]))

                elif not avg_frq:
                    save_dir = op.join(dirs['st_prj'], monkey, condition, event,
                                       norm, '{0}_{1}_tf_mt'.format(f[0], f[1]))

                os.makedirs(save_dir, exist_ok=True)
                fname_mi = op.join(save_dir, 'mi_results.nc'.format(f[0], f[1]))
                fname_pv = op.join(save_dir, 'pv_results.nc'.format(f[0], f[1]))

                if not overwrite and op.exists(fname_mi):
                    mi = xr.load_dataset(fname_mi)
                    pv = xr.load_dataset(fname_pv)

                    ds_mi['times'] = mi['times']
                    ds_pv['times'] = pv['times']

                    ds_mi = mi.update(ds_mi)
                    ds_pv = pv.update(ds_pv)

                ds_mi.to_netcdf(fname_mi)
                ds_pv.to_netcdf(fname_pv)
