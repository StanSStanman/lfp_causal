import os
import os.path as op
import numpy as np
import xarray as xr
from scipy.signal import resample as spr
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
from itertools import product
from lfp_causal.IO import read_session
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo
from lfp_causal.compute_stats import prepare_data


def compute_roi_stats_meso(fname_pow, fname_reg, rois, log_bads, bad_epo,
                           regressor, conditional, mi_type, inference,
                           times=None, freqs=None, avg_freq=True,
                           t_resample=None, f_resample=None):

    mi, pval = [], []
    for r in np.unique(np.array(rois)):
        _fpw, _frg, _frs, _lb, _be = [], [], [], [], []
        for idx, _r in enumerate(rois):
            if r == _r:
                _fpw.append(fname_pow[idx])
                _frg.append(fname_reg[idx])
                _frs.append(rois[idx])
                _lb.append(log_bads[idx])
                _be.append(bad_epo[idx])


        power, regr, cond,\
            t_pnt, freq_b = prepare_data(_fpw, _frg, _lb,
                                         _be, regressor,
                                         cond=conditional, times=times,
                                         freqs=freqs, avg_freq=avg_freq,
                                         t_rsmpl=t_resample,
                                         f_rsmpl=f_resample)

        if mi_type == 'cc':
            regr = [r.astype('float64') for r in regr]
        elif mi_type == 'cd':
            regr = [r.astype('int64') for r in regr]
        elif mi_type == 'ccd':
            regr = [r.astype('float64') for r in regr]
            cond = [c.astype('int64') for c in cond]

        ds_ephy = DatasetEphy(x=power, y=regr, roi=_frs, z=cond, times=t_pnt)

        wf = WfMi(mi_type=mi_type, inference=inference)
        _mi, _pval = wf.fit(ds_ephy, n_perm=1000, n_jobs=-1)

        if not avg_freq:
            mi.assign_coords({'freqs': freq_b})
            pval.assign_coords({'freqs': freq_b})

        mi.append(_mi)
        pval.append(_pval)

    mi = xr.concat(mi, 'roi')
    pval = xr.concat(pval, 'roi')

    return wf, mi, pval


if __name__ == '__main__':
    monkey = 'freddie'
    condition = 'hard'
    event = 'trig_off'
    n_power = '{0}_pow_5_120.nc'.format(event)
    # t_res = 0.001
    times = [(-1.5, 1.3)]
    # freqs = [(5, 120)]
    freqs = [(8, 15), (15, 30), (25, 45), (40, 70), (60, 120)]
    avg_frq = True
    t_resample = None #551
    f_resample = None #80

    epo_dir = '/scratch/rbasanisi/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)
    power_dir = '/scratch/rbasanisi/data/db_lfp/lfp_causal/' \
                '{0}/{1}/pow'.format(monkey, condition)
    regr_dir = '/scratch/rbasanisi/data/db_behaviour/lfp_causal/' \
               '{0}/{1}/regressors'.format(monkey, condition)
    fname_info = '/scratch/rbasanisi/data/db_lfp/lfp_causal/' \
                 '{0}/{1}/files_info.xlsx'.format(monkey, condition)

    # epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
    #           'lfp_causal/{0}/{1}/epo'.format(monkey, condition)
    # power_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
    #             '{0}/{1}/pow'.format(monkey, condition)
    # regr_dir = '/media/jerry/TOSHIBA EXT/data/db_behaviour/lfp_causal/' \
    #            '{0}/{1}/regressors'.format(monkey, condition)
    # fname_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
    #              '{0}/{1}/files_info.xlsx'.format(monkey, condition)

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

    conditionals = [None, None,
                    None, None, None, None,
                    None, None,
                    None, None, None, None, None, None,
                    None, None, None,
                    None, None, None, None,
                    None, None, None,
                    None, None, None,
                    None, None, None,
                    None, None, None,
                    None, None]

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

    # regressors = ['RnR|C']
    # conditionals = [None]
    # mi_type = ['cd']

    inference = ['ffx' for r in regressors]

    fn_pow_list = []
    fn_reg_list = []
    rois = []
    log_bads = []
    bad_epo = []
    rej_files = ['0845', '0847', '0873', '0939', '0945', '1038'] #+ \
                # ['0946', '0948', '0951', '0956', '1135', '1138', '1140',
                #  '1142', '1143', '1144']
    # files = ['0822', '1043', '1191']
    # for d in files:
    for d in os.listdir(power_dir):
        if d in rej_files:
            continue
        if op.isdir(op.join(power_dir, d)):
            fname_power = op.join(power_dir, d, n_power)
            fname_regr = op.join(regr_dir, '{0}.xlsx'.format(d))
            fname_epo = op.join(epo_dir, '{0}_{1}_epo.fif'.format(d, event))

            fn_pow_list.append(fname_power)
            fn_reg_list.append(fname_regr)
            rois.append(read_session(fname_info, d)['sector'].values)

            lb = get_log_bad_epo(fname_epo)
            log_bads.append(lb)

            be = get_ch_bad_epo(monkey, condition, d,
                                fname_info=fname_info)
            bad_epo.append(be)

    mi_results = {}
    pv_results = {}
    for t, f in product(times, freqs):
        # t_pt = np.arange(t[0], t[1] + t_res, t_res).round(5)
        for r, c, m, i in zip(regressors, conditionals, mi_type, inference):
            wf, mi, pvals = compute_roi_stats_meso(fn_pow_list, fn_reg_list,
                                                   rois, log_bads, bad_epo,
                                                   r, c, m, i, t, f, avg_frq,
                                                   t_resample, f_resample)

            mi_results[r] = mi
            pv_results[r] = pvals

        if avg_frq:
            save_dir = op.join('/scratch/rbasanisi/data/stats/lfp_causal/',
                               monkey, condition, event, 'by_roi',
                               '{0}_{1}'.format(f[0], f[1]))

            # save_dir = op.join('/media/jerry/TOSHIBA EXT/data/stats/'
            #                    'lfp_causal/',
            #                    monkey, condition, event, 'by_roi',
            #                    '{0}_{1}'.format(f[0], f[1]))

            os.makedirs(save_dir, exist_ok=True)
            ds_mi = xr.Dataset(mi_results)
            ds_pv = xr.Dataset(pv_results)

            ds_mi.to_netcdf(op.join(save_dir,
                                    'mi_results.nc'.format(f[0], f[1])))
            ds_pv.to_netcdf(op.join(save_dir,
                                    'pv_results.nc'.format(f[0], f[1])))

        elif not avg_frq:
            save_dir = op.join('/scratch/rbasanisi/data/stats/lfp_causal/',
                               monkey, condition, event, 'by_roi',
                               '{0}_{1}_tf'.format(f[0], f[1]))

            # save_dir = op.join('/media/jerry/TOSHIBA EXT/data/stats/'
            #                    'lfp_causal/',
            #                    monkey, condition, event, 'by_roi',
            #                    '{0}_{1}_tf'.format(f[0], f[1]))

            os.makedirs(save_dir, exist_ok=True)
            ds_mi = xr.Dataset(mi_results)
            ds_pv = xr.Dataset(pv_results)

            ds_mi.to_netcdf(op.join(save_dir,
                                    'mi_results.nc'.format(f[0], f[1])))
            ds_pv.to_netcdf(op.join(save_dir,
                                    'pv_results.nc'.format(f[0], f[1])))
