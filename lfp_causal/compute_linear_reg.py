import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
import os
import os.path as op
from lfp_causal.IO import read_session
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo
from lfp_causal.compute_stats import prepare_data

def linear_reg(power, regr, roi=None, times=None, freqs=None):

    # All the needed checks before running analysis
    # Check number of data
    assert len(power) == len(regr), \
        AssertionError('Data must have the same length')
    for p, r in zip(power, regr):
        assert p.shape[0] == len(r), \
            AssertionError('Data must have the same number of trials')
    # Check ROI
    if roi is not None:
        assert len(roi) == len(power), \
            AssertionError('Number of subjects and the length of list of '
                           'roi must be the same')
        for p, ri in zip(power, roi):
            assert p.shape[1] == len(ri), \
                AssertionError('ROI\'s names and ROI\'s number per '
                               'subject must be the same')
    elif roi is None:
        roi = []
        for p in power:
            roi.append(['roi' for _ri in range(p.shape[1])])

    # Take unique roi names
    u_roi = []
    for ri in roi:
        _ri = np.unique(ri)
        for __ri in ri:
            if __ri not in u_roi:
                u_roi.append(__ri)

    res_name = ['R2', 'intercept', 'slope']

    # Divide data by ROI
    results = []
    for ur in u_roi:
        pows = []
        rgs = []
        for p, r, ri in zip(power, regr, roi):
            # Add freq dim if data are avg on freq
            if p.ndim == 3:
                p = np.expand_dims(p, 2)
            # Chose data corresponding to the roi
            if ur in ri:
                i_r = np.where(np.array(ri) == ur)[0][0]
                pows.append(p[:, i_r, :, :])
                rgs.append(r)
        pows = np.concatenate(pows, axis=0)
        rgs = np.concatenate(rgs, axis=0)

        # Perform time frequency analysis (lost roi dim)
        roi_res = np.zeros((3, pows.shape[-2], pows.shape[-1]))
        for i_f, f in enumerate(range(pows.shape[-2])):
            _data = pows[:, f, :]
            # _data = _data.squeeze(1)
            for i_t, t in enumerate(range(_data.shape[-1])):
                _d = _data[:, t].reshape(-1, 1) ##################### CHECK DIMS HERE
                model = LinearRegression()
                model.fit(_d, rgs)
                # Save R2 coeff, intercept and slope
                roi_res[0, i_f, i_t] = model.score(_d, rgs)
                roi_res[1, i_f, i_t] = model.intercept_
                roi_res[2, i_f, i_t] = model.coef_[0]
        results.append(np.expand_dims(roi_res, axis=1))
    results = np.concatenate(results, axis=1)

    if freqs is None or results.shape[2] == 1:
        freqs = list(range(results.shape[-2]))

    results = xr.DataArray(results, coords=[res_name, u_roi, freqs, times],
                         dims=['values', 'rois', 'freqs', 'times'])

    return results


if __name__ == '__main__':

    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_off'
    regressor = 'rpe'
    n_power = '{0}_pow_5_120.nc'.format(event)
    norm = 'relchange'
    # t_res = 0.001
    times = [(-1.5, 1.3)]
    freqs = [(8, 15), (15, 30), (25, 45), (40, 70), (60, 120)]
    times = (-1.5, 1.3)
    freqs = (15, 30)
    avg_frq = True
    t_resample = None #1400
    f_resample = None #80


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
                 '{0}/{1}/files_info.xlsx'.format(monkey, condition)

    fn_pow_list = []
    fn_reg_list = []
    rois = []
    log_bads = []
    bad_epo = []
    rej_files = ['0845', '0847', '0873', '0939', '0945', '1038'] #+ \
                # ['0946', '0948', '0951', '0956', '1135', '1138', '1140',
                #  '1142', '1143', '1144']
    files = ['0822', '1043', '1191']
    for d in files:
    # for d in os.listdir(power_dir):
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

    power, regr, cond,\
        times, freqs = prepare_data(fn_pow_list, fn_reg_list, log_bads,
                                    bad_epo, regressor,
                                    cond=None, times=times,
                                    freqs=freqs, avg_freq=avg_frq,
                                    t_rsmpl=t_resample, f_rsmpl=f_resample,
                                    norm=norm, bline=(-1.8, -1.3))

    lr_results = linear_reg(power, regr, roi=rois, times=times, freqs=freqs)

    if avg_frq:
        # save_dir = op.join('/scratch/rbasanisi/data/stats/lfp_causal/',
        #                    monkey, condition, event, norm,
        #                    '{0}_{1}'.format(freqs[0], freqs[-1]))

        save_dir = op.join('/media/jerry/TOSHIBA EXT/data/stats/lfp_causal/',
                           monkey, condition, event, norm,
                           '{0}_{1}'.format(freqs[0], freqs[-1]))

    elif not avg_frq:
        # save_dir = op.join('/scratch/rbasanisi/data/stats/lfp_causal/',
        #                    monkey, condition, event,
        #                    '{0}_{1}_tf'.format(freqs[0], freqs[-1]))

        save_dir = op.join('/media/jerry/TOSHIBA EXT/data/stats/lfp_causal/',
                           monkey, condition, event, norm,
                           '{0}_{1}_tf'.format(freqs[0], freqs[-1]))

    os.makedirs(save_dir, exist_ok=True)
    # ds_lr = xr.Dataset(lr_results)

    lr_results.to_netcdf(op.join(save_dir,
                         'lr_results.nc'.format(freqs[0], freqs[-1])))
