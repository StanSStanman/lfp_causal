import os
import os.path as op
from itertools import product
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.model_selection import KFold

from research.get_dirs import get_dirs
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo
from lfp_causal.compute_stats import prepare_data


# def prepare_data(power, regs, rois, times, freqs):
#     assert isinstance(power, list)
#     assert isinstance(regs, dict)
#
#     data = []
#     for i, p in enumerate(power):
#         # let's squeeze the freqs dim so far
#         if p.ndim > 3:
#             p = p.squeeze(axis=2)
#         # let's do it case specific so far
#         p = p.mean(axis=-1).squeeze()
#         r = pd.DataFrame.from_dict({_r: regs[_r][i] for _r in regs.keys()})
#
#
#
#
#         # mid = pd.MultiIndex.from_arrays([r[_r] for _r in r.keys()],
#         #                                 names=r.keys())
#         # data.append(xr.DataArray(p, coords=[mid, rois[0], times],
#         #                          dims=['trials', 'rois', 'times']))

def prepare_GLM_data(fname_pow, fname_reg, rois, log_bads, bad_epo,
                     regressor, conditional, mi_type, inference,
                     times=None, freqs=None, avg_freq=True,
                     t_resample=None, f_resample=None, norm=None):

    power, regr, cond,\
        times, freqs = prepare_data(fname_pow, fname_reg, log_bads,
                                    bad_epo, regressor,
                                    cond=conditional, times=times,
                                    freqs=freqs, avg_freq=avg_freq,
                                    t_rsmpl=t_resample, f_rsmpl=f_resample,
                                    norm=norm, bline=(-.6, -0.1),
                                    fbl='cue_on_pow_beta_mt.nc')
    return power, regr


def compute_linear_regression(power, regs, modality='linear'):
    assert isinstance(power, list)
    assert isinstance(regs, dict)

    if modality == 'linear':
        regression = LinearRegression
    elif modality == 'ridge':
        regression = Ridge

    # ds = xr.concat(ds.x, dim='trials')

    # for t in ds.times:
    #     ds.loc[{'times': 0}].data

    x, y = [], []
    for i, p in enumerate(power):
        # let's squeeze the freqs dim so far
        if p.ndim > 3:
            p = p.squeeze(axis=2)
        # let's do it case specific so far
        p = p.mean(axis=-1).squeeze()
        r = pd.DataFrame.from_dict({_r: regs[_r][i] for _r in regs.keys()})
        # r.insert(loc=0, column='Constant', value=np.ones(len(r)))

        x.append(r)
        y.append(p)

    x = pd.concat(x, ignore_index=True)
    y = np.concatenate(y)

    model = regression()
    model.fit(x, y)
    score = model.score(x, y)
    params = np.append(model.intercept_, model.coef_)
    # params = model.coef_
    pred = model.predict(x)

    nx = np.append(np.ones((len(x), 1)), x, axis=1)
    # nx = np.array(x)
    MSE = (sum((y - pred) ** 2)) / (len(nx) - len(nx[0]))

    var_b = MSE * (np.linalg.inv(np.dot(nx.T, nx)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(nx)-len(nx[0]))))
                for i in ts_b]

    np.set_printoptions(precision=6)
    table = pd.DataFrame()
    table['Regs'] = ['Costant'] + list(x.keys())
    table['Coefficient'] = np.array(params).round(5)
    table['Standard Error'] = np.array(sd_b).round(5)
    table['t values'] = np.array(ts_b).round(5)
    table['p values'] = np.array(p_values).round(5)
    table['R squared'] = np.array([score] * len(p_values)).round(5)

    table.set_index('Regs', inplace=True)

    print(table)

    return table


def corrected_linear_regression(power, regs, modality='linear'):
    assert isinstance(power, list)
    assert isinstance(regs, dict)

    if modality == 'linear':
        regression = LinearRegression
    elif modality == 'ridge':
        regression = Ridge

    coeff = []
    for i, p in enumerate(power):
        # let's squeeze the freqs dim so far
        if p.ndim > 3:
            p = p.squeeze(axis=2)
        # let's do it case specific so far
        y = p.mean(axis=-1).squeeze()
        x = pd.DataFrame.from_dict({_r: regs[_r][i] for _r in regs.keys()})

        model = regression()
        model.fit(x, y)
        # score = model.score(x, y)
        # params = np.append(model.intercept_, model.coef_)
        # pred = model.predict(x)
        coeff.append(model.coef_)

    colnames = list(x.keys())
    coeff = np.vstack(coeff)
    # dfcoeff = pd.DataFrame(data=coeff, index=range(coeff.shape[0]),
    #                        columns=['q_rpe', 'q_absrpe', 'RT',
    #                                 'MT', 'Actions'])
    dfcoeff = pd.DataFrame(data=coeff, index=range(coeff.shape[0]),
                           columns=colnames)
    # sns.violinplot(data=dfcoeff, palette="Set2", scale='count',
    #                inner="point", width=.8)
    # plt.show()
    # sns.boxplot(data=dfcoeff, palette="Set2", width=.8)
    # plt.show()
    stat, pval = stats.ttest_1samp(coeff, popmean=0., axis=0)

    #
    # x = pd.concat(x, ignore_index=True)
    # y = np.concatenate(y)
    #
    # model = regression()
    # model.fit(x, y)
    # score = model.score(x, y)
    # params = np.append(model.intercept_, model.coef_)
    # pred = model.predict(x)

    # nx = np.append(np.ones((len(x), 1)), x, axis=1)
    # MSE = (sum((y - pred) ** 2)) / (len(nx) - len(nx[0]))
    #
    # var_b = MSE * (np.linalg.inv(np.dot(nx.T, nx)).diagonal())
    # sd_b = np.sqrt(var_b)
    # ts_b = params / sd_b
    #
    # p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(nx) - len(nx[0]))))
    #             for i in ts_b]
    #
    # np.set_printoptions(precision=6)
    table = pd.DataFrame()
    table['Regs'] = list(x.keys())
    # table['Coefficient'] = np.array(params).round(5)
    # table['Standard Error'] = np.array(sd_b).round(5)
    table['t values'] = np.array(stat).round(5)
    table['p values'] = np.array(pval).round(5)
    # table['R squared'] = np.array([score] * len(p_values)).round(5)

    table.set_index('Regs', inplace=True)

    print(table)

    return table


def corrected_linear_regression_sm(power, regs, modality='OLS'):
    import statsmodels.api as sm
    assert isinstance(power, list)
    assert isinstance(regs, dict)

    if modality == 'OLS':
        model = sm.OLS
    elif modality == 'GLM':
        model = sm.GLM

    coeff = []
    for i, p in enumerate(power):
        # let's squeeze the freqs dim so far
        if p.ndim > 3:
            p = p.squeeze(axis=2)
        # let's do it case specific so far
        y = p.mean(axis=-1).squeeze()
        x = pd.DataFrame.from_dict({_r: regs[_r][i] for _r in regs.keys()})

        results = model(y, sm.add_constant(x, has_constant='add')).fit()
        coeff.append(results.params.values)
        colname = list(results.params.keys())
        # score = model.score(x, y)
        # params = np.append(model.intercept_, model.coef_)
        # pred = model.predict(x)
        # coeff.append(model.coef_)

    coeff = np.vstack(coeff)
    # dfcoeff = pd.DataFrame(data=coeff, index=range(coeff.shape[0]),
    #                        columns=['q_rpe', 'q_absrpe', 'RT',
    #                                 'MT', 'Actions'])
    dfcoeff = pd.DataFrame(data=coeff, index=range(coeff.shape[0]),
                           columns=colname)
    # sns.violinplot(data=dfcoeff, palette="Set2", scale='count',
    #                inner="point", width=.8)
    # plt.show()
    # sns.boxplot(data=dfcoeff, palette="Set2", width=.8)
    # plt.show()
    stat, pval = stats.ttest_1samp(coeff, popmean=0., axis=0)

    table = pd.DataFrame()
    table['Regs'] = colname
    # table['Coefficient'] = np.array(params).round(5)
    # table['Standard Error'] = np.array(sd_b).round(5)
    table['t values'] = np.array(stat).round(5)
    table['p values'] = np.array(pval).round(5)
    # table['R squared'] = np.array([score] * len(p_values)).round(5)

    table.set_index('Regs', inplace=True)

    print(table)

    return table


def compute_cross_validation(power, regs, modality='linear'):
    assert isinstance(power, list)
    assert isinstance(regs, dict)

    if modality == 'linear':
        regr = LinearRegression()
    elif modality == 'ridge':
        regr = Ridge()
    cv = KFold(n_splits=10)

    # ds = xr.concat(ds.x, dim='trials')

    # for t in ds.times:
    #     ds.loc[{'times': 0}].data

    x, y = [], []
    for i, p in enumerate(power):
        # let's squeeze the freqs dim so far
        if p.ndim > 3:
            p = p.squeeze(axis=2)
        # let's do it case specific so far
        p = p.mean(axis=-1).squeeze()
        r = pd.DataFrame.from_dict({_r: regs[_r][i] for _r in regs.keys()})
        # r.insert(loc=0, column='Constant', value=np.ones(len(r)))

        x.append(r)
        y.append(p)

    x = pd.concat(x, ignore_index=True)
    y = np.concatenate(y)

    ev = []
    for k in x.keys():
        _ev = cross_val_score(regr, x[k].values.reshape(-1, 1), y, cv=cv,
                              scoring="r2")
        ev.append(_ev)
    ev = np.vstack(ev).T

    evcoeff = pd.DataFrame(data=ev, index=range(ev.shape[0]),
                           columns=list(regs.keys()))

    sns.boxplot(data=evcoeff, palette="Set2", width=.8)
    plt.show()

    return evcoeff


def corrected_cross_validation(power, regs, modality='linear'):
    assert isinstance(power, list)
    assert isinstance(regs, dict)

    if modality == 'linear':
        regr = LinearRegression()
    elif modality == 'ridge':
        regr = Ridge()
    cv = KFold(n_splits=3)

    ev = {k: [] for k in regs.keys()}
    for i, p in enumerate(power):
        # let's squeeze the freqs dim so far
        if p.ndim > 3:
            p = p.squeeze(axis=2)
        # let's do it case specific so far
        y = p.mean(axis=-1).squeeze()
        x = pd.DataFrame.from_dict({_r: regs[_r][i] for _r in regs.keys()})

        for k in x.keys():
            # _ev = cross_val_score(regr, x[k].values.reshape(-1, 1), y, cv=cv,
            #                       scoring="explained_variance")
            _ev = cross_val_score(regr, y.reshape(-1, 1), x[k].values, cv=cv,
                                  scoring="r2")
            ev[k].append(_ev)

    for k in regs.keys():
        ev[k] = np.concatenate(ev[k])
    # ev = np.vstack(ev).T

    evcoeff = pd.DataFrame(ev)
    # evcoeff = pd.DataFrame(data=ev, index=range(ev.shape[0]),
    #                        columns=list(regs.keys()))
    # sns.violinplot(data=dfcoeff, palette="Set2", scale='count',
    #                inner="point", width=.8)
    # plt.show()
    sns.boxplot(data=evcoeff, palette="Set2", width=.8)
    plt.show()
    # stat, pval = stats.ttest_1samp(coeff, popmean=0., axis=0)

    #
    # x = pd.concat(x, ignore_index=True)
    # y = np.concatenate(y)
    #
    # model = regression()
    # model.fit(x, y)
    # score = model.score(x, y)
    # params = np.append(model.intercept_, model.coef_)
    # pred = model.predict(x)

    # nx = np.append(np.ones((len(x), 1)), x, axis=1)
    # MSE = (sum((y - pred) ** 2)) / (len(nx) - len(nx[0]))
    #
    # var_b = MSE * (np.linalg.inv(np.dot(nx.T, nx)).diagonal())
    # sd_b = np.sqrt(var_b)
    # ts_b = params / sd_b
    #
    # p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(nx) - len(nx[0]))))
    #             for i in ts_b]
    #
    # np.set_printoptions(precision=6)
    # table = pd.DataFrame()
    # table['Regs'] = list(x.keys())
    # # table['Coefficient'] = np.array(params).round(5)
    # # table['Standard Error'] = np.array(sd_b).round(5)
    # table['t values'] = np.array(stat).round(5)
    # table['p values'] = np.array(pval).round(5)
    # # table['R squared'] = np.array([score] * len(p_values)).round(5)
    #
    # table.set_index('Regs', inplace=True)
    #
    # print(table)

    return evcoeff


if __name__ == '__main__':
    from lfp_causal import MCH, PRJ
    dirs = get_dirs(MCH, PRJ)

    monkeys = ['freddie', 'teddy']
    conditions = ['easy', 'hard']
    event = 'trig_off'
    norm = 'fbline_relchange'
    n_power = '{0}_pow_beta_mt.nc'.format(event)

    times = [(0., .8)]
    # times = [(0., .2), (.2, .4), (.4, .6), (.6, .8)]
    freqs = [(15, 35)]

    avg_frq = False
    t_resample = None #1400
    f_resample = None #80
    overwrite = True

    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']

    # regressors = ['Reward', 'q_rpe', 'q_absrpe', 'RT', 'MT', 'q_dP', 'Actions']
    # conditionals = [None, None, None, None, None, None, None]
    # mi_type = ['cd', 'cc', 'cc', 'cc', 'cc', 'cc', 'cd']
    # inference = ['ffx' for r in regressors]
    # regressors = ['q_rpe', 'q_absrpe', 'RT', 'MT']
    # conditionals = [None, None, None, None]
    # mi_type = ['cc', 'cc', 'cc', 'cc']
    # inference = ['ffx' for r in regressors]
    regressors = ['q_rpe', 'q_absrpe', 'RT', 'MT', 'Actions']
    conditionals = [None, None, None, None, None]
    mi_type = ['cc', 'cc', 'cc', 'cc', 'cd']
    inference = ['ffx' for r in regressors]
    # regressors = ['q_rpe', 'q_absrpe', 'AT', 'Actions']
    # conditionals = [None, None, None, None]
    # mi_type = ['cc', 'cc', 'cc', 'cd']
    # inference = ['ffx' for r in regressors]
    regressors = ['q_rpe', 'q_absrpe', 'RT', 'MT']
    conditionals = [None, None, None, None]
    mi_type = ['cc', 'cc', 'cc', 'cc']
    inference = ['ffx' for r in regressors]

    rej_files = []
    rej_files += ['1204', '1217', '1231', '0944',  # Bad sessions
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
                  '0291', '0368', '0743']

    rej_files += ['0231', '0272', '0274', '0666', '0941', '0855', '0722',
                  '0725', '1397', '1398', '1701']
    # files = ['0832', '0822', '1043', '1191']
    # for d in files:

    ##############################
    for monkey in monkeys:
        fn_pow_list = []
        fn_reg_list = []
        rois = []
        log_bads = []
        bad_epo = []
    ##############################

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
                if op.isdir(op.join(power_dir, d)):
                    fname_power = op.join(power_dir, d, n_power)
                    fname_regr = op.join(regr_dir, '{0}.xlsx'.format(d))
                    fname_epo = op.join(epo_dir,
                                        '{0}_{1}_epo.fif'.format(d, event))

                    fn_pow_list.append(fname_power)
                    fn_reg_list.append(fname_regr)
                    # rois.append(read_session(fname_info, d)['sector'].values)
                    # For noroi condition:
                    # rois.append([str(d)])
                    # For uniroi condition:
                    rois.append(['all_rois'])

                    lb = get_log_bad_epo(fname_epo)
                    log_bads.append(lb)

                    be = get_ch_bad_epo(monkey, condition, d,
                                        fname_info=fname_info)
                    bad_epo.append(be)

            # mi_results = {}
            # pv_results = {}
        for t, f in product(times, freqs):
            power, regrs = prepare_GLM_data(fn_pow_list, fn_reg_list,
                                            rois, log_bads, bad_epo,
                                            regressors, conditionals,
                                            mi_type, inference,
                                            t, f, avg_frq,
                                            t_resample, f_resample,
                                            norm)

            mode = 'linear'
            table = compute_linear_regression(power, regrs, modality=mode)
            table2 = corrected_linear_regression(power, regrs, modality=mode)
            table3 = corrected_linear_regression_sm(power, regrs,
                                                    modality='GLM')
            table4 = compute_cross_validation(power, regrs)
            table5 = corrected_cross_validation(power, regrs)

            # path = '/home/jerry/Immagini/new_imgs/' \
            #        'MLR_{0}_{1}_{2}.xlsx'.format(monkey, mode, t[0])
            path = '/home/jerry/Immagini/new_imgs/' \
                   'MLR_{0}_{1}_{2}.xlsx'.format(monkey, mode, t[0])
            print(path)
            # writer = pd.ExcelWriter(path, engine='xlsxwriter')
            # table.to_excel(writer, sheet_name='Table_1')
            # table2.to_excel(writer, sheet_name='Table_2')
            # table3.to_excel(writer, sheet_name='Table_3')
            # writer.save()
            # writer.close()
            # ##################

    # from lfp_causal.IO import read_sector
    # for sect in sectors:
    #
    #     ##############################
    #     for monkey in monkeys:
    #         fn_pow_list = []
    #         fn_reg_list = []
    #         rois = []
    #         log_bads = []
    #         bad_epo = []
    #     ##############################
    #
    #         for condition in conditions:
    #             # fn_pow_list = []
    #             # fn_reg_list = []
    #             # rois = []
    #             # log_bads = []
    #             # bad_epo = []
    #
    #             epo_dir = dirs['epo'].format(monkey, condition)
    #             power_dir = dirs['pow'].format(monkey, condition)
    #             regr_dir = dirs['reg'].format(monkey, condition)
    #             fname_info = op.join(dirs['ep_cnds'].format(monkey, condition),
    #                                  'files_info.xlsx')
    #             rec_info = op.join(dirs['ep_cnds'].format(monkey, condition),
    #                                'files_info.xlsx')
    #
    #             fid = read_sector(rec_info, sect)
    #             fid = fid[fid['quality'] <= 3]
    #
    #             for d in fid['file']:
    #                 if d in rej_files:
    #                     continue
    #                 if op.isdir(op.join(power_dir, d)):
    #                     fname_power = op.join(power_dir, d, n_power)
    #                     fname_regr = op.join(regr_dir, '{0}.xlsx'.format(d))
    #                     fname_epo = op.join(epo_dir,
    #                                         '{0}_{1}_epo.fif'.format(d, event))
    #
    #                     fn_pow_list.append(fname_power)
    #                     fn_reg_list.append(fname_regr)
    #                     # rois.append(read_session(fname_info, d)['sector'].values)
    #                     # For noroi condition:
    #                     # rois.append([str(d)])
    #                     # For uniroi condition:
    #                     rois.append(['all_rois'])
    #
    #                     lb = get_log_bad_epo(fname_epo)
    #                     log_bads.append(lb)
    #
    #                     be = get_ch_bad_epo(monkey, condition, d,
    #                                         fname_info=fname_info)
    #                     bad_epo.append(be)
    #
    #             # mi_results = {}
    #             # pv_results = {}
    #         for t, f in product(times, freqs):
    #             power, regrs = prepare_GLM_data(fn_pow_list, fn_reg_list,
    #                                             rois, log_bads, bad_epo,
    #                                             regressors, conditionals,
    #                                             mi_type, inference,
    #                                             t, f, avg_frq,
    #                                             t_resample, f_resample,
    #                                             norm)
    #
    #             mode = 'linear'
    #             table = compute_linear_regression(power, regrs, modality=mode)
    #             table2 = corrected_linear_regression(power, regrs, modality=mode)
    #
    #             path = '/home/jerry/Immagini/new_imgs/' \
    #                    'MLR_{0}_{1}_{2}.xlsx'.format(monkey, mode, sect)
    #             writer = pd.ExcelWriter(path, engine='xlsxwriter')
    #             table.to_excel(writer, sheet_name='Table_1')
    #             table2.to_excel(writer, sheet_name='Table_2')
    #             writer.save()
    #             writer.close()
    #             # ##################