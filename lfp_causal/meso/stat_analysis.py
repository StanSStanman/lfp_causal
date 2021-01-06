import os
import os.path as op
import numpy as np
import xarray as xr
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
from itertools import product
from lfp_causal.IO import read_session
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo
from lfp_causal.compute_stats import prepare_data


def compute_stats_meso(fname_pow, fname_reg, rois, log_bads, bad_epo,
                       regressor, conditional, mi_type, inference,
                       times=None, freqs=None, avg_freq=True):

    power, regr, cond = prepare_data(fname_pow, fname_reg, log_bads,
                                     bad_epo, regressor,
                                     cond=conditional, times=times,
                                     freqs=freqs, avg_freq=avg_freq)

    ds_ephy = DatasetEphy(x=power, y=regr, roi=rois, z=cond, times=times)

    wf = WfMi(mi_type=mi_type, inference=inference)
    mi, pval = wf.fit(ds_ephy, n_perm=1000)

    return wf, mi, pval


if __name__ == '__main__':
    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_off'
    n_power = '{0}_pow_5_70.nc'.format(event)
    t_res = 0.0005
    times = [(-1., 1.3)]
    freqs = [(8, 15), (15, 30), (25, 50), (40, 70)]

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
                  '#R', '#nR', '#R|C', '#nR|C', '#R|nC', '#nR|nC',
                  'learn_5t', 'learn_2t', 'early_late_cons',
                  'P(R|C)', 'P(R|nC)', 'P(R|Cho)',
                  'dP', 'log_dP', 'delta_dP',
                  'surprise', 'surprise_bayes', 'rpe']

    conditionals = [None, None,
                    None, None, None, None,
                    None, None, None, None, None, None,
                    None, None, None,
                    None, None, None,
                    None, None, None,
                    None, None, None, ]

    mi_type = ['cd', 'cd',
               'cd', 'cd', 'cd', 'cd',
               'cc', 'cc', 'cc', 'cc', 'cc', 'cc',
               'cd', 'cd', 'cd',
               'cc', 'cc', 'cc',
               'cc', 'cc', 'cc',
               'cc', 'cc', 'cc', ]

    inference = ['ffx' for r in regressors]

    fn_pow_list = []
    fn_reg_list = []
    rois = []
    log_bads = []
    bad_epo = []
    # files = ['0822', '1043', '1191']
    # for d in files:
    for d in os.listdir(power_dir)[:5]:
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
        t_pt = np.arange(t[0], t[1] + t_res, t_res).round(5)
        for r, c, m, i in zip(regressors, conditionals, mi_type, inference):
            wf, mi, pvals = compute_stats_meso(fn_pow_list, fn_reg_list, rois,
                                               log_bads, bad_epo,
                                               r, c, m, i, t_pt, f, True)

            mi_results[r] = mi
            pv_results[r] = pvals

        os.makedirs('/scratch/rbasanisi/data/stats/lfp_causal/'
                    '{0}_{1}/'.format(f[0], f[1]), exist_ok=True)
        ds_mi = xr.Dataset(mi_results)
        ds_pv = xr.Dataset(pv_results)

        ds_mi.to_netcdf('/scratch/rbasanisi/data/stats/lfp_causal/'
                        '{0}_{1}/mi_results.nc').format(f[0], f[1])
        ds_pv.to_netcdf('/scratch/rbasanisi/data/stats/lfp_causal/'
                        '{0}_{1}/pv_results.nc').format(f[0], f[1])

        # os.makedirs('/media/jerry/TOSHIBA EXT/data/stats/lfp_causal/'
        #             '{0}_{1}/'.format(f[0], f[1]),
        #             exist_ok=True)
        # ds_mi = xr.Dataset(mi_results)
        # ds_pv = xr.Dataset(pv_results)
        #
        # ds_mi.to_netcdf('/media/jerry/TOSHIBA EXT/data/stats/lfp_causal/'
        #                 '{0}_{1}/mi_results.nc').format(f[0], f[1])
        # ds_pv.to_netcdf('/media/jerry/TOSHIBA EXT/data/stats/lfp_causal/'
        #                 '{0}_{1}/pv_results.nc').format(f[0], f[1])
