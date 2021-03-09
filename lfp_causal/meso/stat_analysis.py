import os
import os.path as op
import xarray as xr
import numpy as np
from frites.dataset import DatasetEphy
from frites.workflow import WfMi
from itertools import product
from research.get_dirs import get_dirs
from lfp_causal.IO import read_session
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo
from lfp_causal.compute_stats import prepare_data
# os.system("taskset -p 0xff %d" % os.getpid())
from lfp_causal.profiling import (RepeatedTimer, memory_usage, cpu_usage)
import time
import json


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
                                    fbl='cue_on_pow_8_120_mt.nc')

    ###########################################################################
    # mu = RepeatedTimer(1, memory_usage)
    # cu = RepeatedTimer(1, cpu_usage)
    ###########################################################################

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
            mi.assign_coords({'freqs': freqs})
            pval.assign_coords({'freqs': freqs})

        mi_results[_r] = mi
        pv_results[_r] = pval
        if _inf == 'rfx':
            conj_ss_results[_r] = conj_ss
            conj_results[_r] = conj

    ###########################################################################
    # mu.stop()
    # cu.stop()
    # ftm = time.strftime('%d%m%y%H%M%S', time.localtime())
    # running = 'fts0_fr_ea'
    # m_out_dir = op.join('/home', 'rbasanisi', 'profiling', 'memory', running)
    # c_out_dir = op.join('/home', 'rbasanisi', 'profiling', 'cpu', running)
    # for d in [m_out_dir, c_out_dir]:
    #     os.makedirs(d, exist_ok=True)
    # m_out_file = op.join(m_out_dir, 'memory_test_{0}.json'.format(ftm))
    # c_out_file = op.join(c_out_dir, 'cpu_test_{0}.json'.format(ftm))
    # for jfn, td in zip([m_out_file, c_out_file],
    #                    [memory_usage(), cpu_usage()]):
    #     with open(jfn, 'w') as jf:
    #         json.dump(td, jf)
    ###########################################################################

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
    event = 'trig_off'
    norm = 'fbline_relchange'
    n_power = '{0}_pow_8_120_mt.nc'.format(event)
    times = [(-1.5, 1.3)]
    # times = [(-1.5, 1.7)]
    # freqs = [(5, 120)]
    # freqs = [(8, 15), (15, 30), (25, 45), (40, 70), (60, 120)]
    freqs = [(8, 12), (15, 35), (40, 65), (70, 120)]
    avg_frq = True
    t_resample = None #1400
    f_resample = None #80
    overwrite = True

    for condition in conditions: # Try to compute multiple conditions

        regressors = ['Correct', 'Reward',
                      'is_R|C', 'is_nR|C', 'is_R|nC', 'is_nR|nC',
                      'RnR|C', 'RnR|nC',
                      '#R', '#nR', '#R|C', '#nR|C', '#R|nC', '#nR|nC',
                      'learn_5t', 'learn_2t', 'early_late_cons',
                      'P(R|C)', 'P(R|nC)', 'P(R|Cho)', 'P(R|A)',
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
                   'cc', 'cc', 'cc',
                   'cc', 'cc', 'cc', 'cc',
                   'cc', 'cc', 'cc',
                   'cc', 'cc', 'cc',
                   'cc', 'cc']

        # regressors = ['act_surp_bayes']
        # conditionals = [None]
        # mi_type = ['cc']

        inference = ['ffx' for r in regressors]

        fn_pow_list = []
        fn_reg_list = []
        rois = []
        log_bads = []
        bad_epo = []
        # rej_files = ['0845', '0847', '0873', '0939', '0945', '1038', '1204',
        #              '1217'] + \
        #             ['0944', '0967', '0969', '0967', '0970', '0971', '1139',
        #              '1145', '1515', '1701']
                    # ['0946', '0948', '0951', '0956', '1135', '1138', '1140',
                    #  '1142', '1143', '1144']
        rej_files = []
        rej_files += ['1204', '1217', '1231', '0944', # Bad sessions
                      '0845', '0847', '0939', '0946', '0963', '1036', '1231',
                      '1233', '1234', '1514', '1699',
                      '0940', '0944', '0964', '0967', '0969', '0970', '0971',
                      '0977', '0985', '1280']
        rej_files += ['0415', '0449', '0450',
                      '0416']
        # files = ['0832', '0822', '1043', '1191']
        # for d in files:
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
                    rois.append(read_session(fname_info, d)['sector'].values)

                    lb = get_log_bad_epo(fname_epo)
                    log_bads.append(lb)

                    be = get_ch_bad_epo(monkey, condition, d,
                                        fname_info=fname_info)
                    bad_epo.append(be)

        mi_results = {}
        pv_results = {}
        for t, f in product(times, freqs):
            ds_mi, ds_pv = compute_stats_meso(fn_pow_list, fn_reg_list, rois,
                                              log_bads, bad_epo,
                                              regressors, conditionals,
                                              mi_type, inference,
                                              t, f, avg_frq,
                                              t_resample, f_resample,
                                              norm)
            ##################
            if len(monkeys) > 1:
                monkey = 'freted'
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

                ds_mi = xr.merge([mi, ds_mi])
                ds_pv = xr.merge([pv, ds_pv])

            ds_mi.to_netcdf(fname_mi)
            ds_pv.to_netcdf(fname_pv)
