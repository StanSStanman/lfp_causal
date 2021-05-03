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
from lfp_causal.compute_stats import prepare_data
# os.system("taskset -p 0xff %d" % os.getpid())
# from lfp_causal.profiling import (RepeatedTimer, memory_usage, cpu_usage)
# import time
# import json


def compute_stats_meso(fname_pow, fname_reg, rois, log_bads, bad_epo,
                       regressor, conditional, mi_type, inference,
                       times=None, freqs=None, avg_freq=True,
                       t_resample=None, f_resample=None, norm=None):

    power, regr, cond,\
        times, freqs = prepare_data(fname_pow, fname_reg, log_bads,
                                    bad_epo, regressor,
                                    cond=conditional, times=times,
                                    freqs=freqs, avg_freq=avg_freq,
                                    t_rsmpl=t_resample, f_rsmpl=f_resample,
                                    norm=norm, bline=(-.55, -0.05),
                                    fbl='cue_on_pow_8_120.nc')

    # for i, p in enumerate(power):
    #     power[i] = p[:25, :, :]
    # for k in regr.keys():
    #     for i, r in enumerate(regr[k]):
    #         regr[k][i] = r[:25]
    # for k in cond.keys():
    #     if cond[k] is not None:
    #         for i, c in enumerate(cond[k]):
    #             cond[k][i] = c[:25]

    mi_results = {}
    pv_results = {}
    conj_ss_results = {}
    conj_results = {}
    for _r, _c, _mt, _inf in zip(regressor, conditional, mi_type, inference):
        if _mt == 'cc':
            regr[_r] = [r.astype('float32') for r in regr[_r]]
        elif _mt == 'cd':
            regr[_r] = [r.astype('int32') for r in regr[_r]]
        elif _mt == 'ccd':
            regr[_r] = [r.astype('float32') for r in regr[_r]]
            cond[_r] = [c.astype('int32') for c in cond[_r]]

        ds_ephy = DatasetEphy(x=power.copy(), y=regr[_r], roi=rois,
                              z=cond[_r], times=times)

        wf = WfMi(mi_type=_mt, inference=_inf, kernel=np.hanning(20))
        mi, pval = wf.fit(ds_ephy, n_perm=1000, n_jobs=-1)
        mi['times'] = times
        pval['times'] = times

        if _inf == 'rfx':
            conj_ss, conj = wf.conjunction_analysis(ds_ephy)

        if not avg_freq:
            # mi.assign_coords({'freqs': freqs})
            # pval.assign_coords({'freqs': freqs})
            mi.assign_coords({'supp': freqs})
            pval.assign_coords({'supp': freqs})


        mi_results[_r] = mi
        pv_results[_r] = pval

        # mi_results['q_rpw_act'] = mi
        # pv_results['q_rpw_act'] = pval

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

    monkeys = ['freddie', 'teddy']
    conditions = ['easy', 'hard']
    # conditions = ['cued']
    event = 'trig_off'
    norm = 'fbline_relchange'
    n_power = '{0}_pow_8_120.nc'.format(event)
    # times = [(-1.5, 1.3)]
    times = [(0., .8)]
    freqs = [(8, 80)]
    # freqs = [(8, 15), (15, 30), (25, 45), (40, 70), (60, 120)]
    # freqs = [(8, 12), (15, 35), (40, 65), (70, 120)]
    avg_frq = False
    t_resample = None #1400
    f_resample = None #80
    overwrite = True

    fname_clus = op.join(dirs['ep_sbjs'], 'clusters_info.xlsx')

    regressors = ['Correct', 'Reward',
                  'is_R|C', 'is_nR|C', 'is_R|nC', 'is_nR|nC',
                  'RnR|C', 'RnR|nC',
                  '#R', '#nR', '#R|C', '#nR|C', '#R|nC', '#nR|nC',
                  'learn_5t', 'learn_2t', 'early_late_cons',
                  'P(R|C)', 'P(R|nC)', 'P(R|Cho)', 'P(R|A)',
                  'pra_mean',
                  'dP', 'log_dP', 'delta_dP',
                  'surprise', 'surprise_bayes', 'act_surp_bayes', 'rpe',
                  'q_pcorr', 'q_pincorr', 'q_dP',
                  'q_entropy', 'q_rpe', 'q_absrpe',
                  'q_shann_surp', 'q_bayes_surp']

    conditionals = [None, None,
                    None, None, None, None,
                    None, None,
                    None, None, None, None, None, None,
                    None, None, None,
                    None, None, None, None,
                    None,
                    None, None, None,
                    None, None, None, None,
                    None, None, None,
                    None, None, None,
                    None, None]

    mi_type = ['cd', 'cd',
               'cd', 'cd', 'cd', 'cd',
               'cd', 'cd',
               'cc', 'cc', 'cc', 'cc', 'cc', 'cc',
               'cd', 'cd', 'cd',
               'cc', 'cc', 'cc', 'cc',
               'cc',
               'cc', 'cc', 'cc',
               'cc', 'cc', 'cc', 'cc',
               'cc', 'cc', 'cc',
               'cc', 'cc', 'cc',
               'cc', 'cc']

    regressors = ['Reward', 'q_rpe']
    conditionals = [None, None]
    mi_type = ['cd', 'cc']

    inference = ['ffx' for r in regressors]

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
                  '0291', '0368', '0743',

                  '0722', '0725']

    ##############################
    for monkey in monkeys:
        fn_pow_list = []
        fn_reg_list = []
        rois = []
        log_bads = []
        bad_epo = []
    ##############################
        fn_cl = fname_clus.format(monkey)
        info = pd.read_excel(fn_cl, sheet_name='easy_hard', index_col=None,
                             dtype={'file': str, 'KM_pos_cluster': str})

        for condition in conditions:
            # fn_pow_list = []
            # fn_reg_list = []
            # rois = []
            # log_bads = []
            # bad_epo = []

            epo_dir = dirs['epo'].format(monkey, condition)
            power_dir = dirs['pow'].format(monkey, condition)
            regr_dir = dirs['reg'].format(monkey, condition)
            fname_info = op.join(dirs['ep_cnds'].format(monkey, condition),
                                 'files_info.xlsx')

            for d in os.listdir(power_dir):
                if d in rej_files:
                    continue
                elif d not in info['file'].values:
                    continue

                if op.isdir(op.join(power_dir, d)):
                    fname_power = op.join(power_dir, d, n_power)
                    fname_regr = op.join(regr_dir, '{0}.xlsx'.format(d))
                    fname_epo = op.join(epo_dir,
                                        '{0}_{1}_epo.fif'.format(d, event))

                    fn_pow_list.append(fname_power)
                    fn_reg_list.append(fname_regr)
                    # rois.append(read_session(fname_info, d)['sector'].values)
                    cl = info['KM_pos_cluster'][info['file'] == d].values
                    rois.append(cl)

                    lb = get_log_bad_epo(fname_epo)
                    log_bads.append(lb)

                    be = get_ch_bad_epo(monkey, condition, d,
                                        fname_info=fname_info)
                    bad_epo.append(be)

            # mi_results = {}
            # pv_results = {}
        for t, f in product(times, freqs):
            ds_mi, ds_pv = compute_stats_meso(fn_pow_list, fn_reg_list,
                                              rois, log_bads, bad_epo,
                                              regressors, conditionals,
                                              mi_type, inference,
                                              t, f, avg_frq,
                                              t_resample, f_resample,
                                              norm)
            ##################
            # if len(monkeys) > 1:
            #     monkey = 'freted'

            # if 'easy' in condition:
            #     condition = 'easy_25'
            # elif 'hard' in condition:
            #     condition = 'hard_25'
            # elif 'cued' in condition:
            #     condition = 'cued_25'
            condition = 'eaha_25_clusters_tf'
            ##################

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

            print('Saved', fname_mi)
            print('Saved', fname_pv)
