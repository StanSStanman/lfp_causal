import os
import os.path as op
import mne
import numpy as np
import pandas as pd
import xarray as xr
from frites.dataset import DatasetEphy
from frites.workflow import WfMi

from research.get_dirs import get_dirs
from lfp_causal.compute_power import normalize_power
from lfp_causal.IO import read_sector
from lfp_causal.compute_bad_epochs import get_ch_bad_epo


def prepare_data(epochs, reg_fname, regressors, bads, times=None,
                 norm='cue_on', bline=None):
    d_phys = []
    d_regr = []
    for p, r, b in zip(epochs, reg_fname, bads):
        print('Loading', p)
        epo = mne.read_epochs(p)
        if times is not None:
            epo.crop(*times)
        t_points = epo.times
        reg = pd.read_excel(r)[regressors]
        epo = epo.drop(b)
        br = [i for i, dl in enumerate(epo.drop_log) if len(dl) != 0]
        reg.drop(index=br, inplace=True)
        epo = epo._data.squeeze()

        if norm is not None:
            bln_fname = p.replace('trig_off', norm)
            bln = mne.read_epochs(bln_fname)
            bln.crop(*bline)
            bln.drop(b)
            bln = bln._data.squeeze()
            epo = (epo - bln.mean(1, keepdims=True)) / \
                   bln.mean(1, keepdims=True)

        d_phys.append(np.expand_dims(epo, 1))
        d_regr.append(reg)

    return d_phys, d_regr, t_points


def compute_stats(epochs, reg_fname, regressors, bads, rois, mi_type,
                  inference, times=None, norm='cue_on'):

    epd, rgd, tp = prepare_data(epochs, reg_fname, regressors, bads, times,
                                norm, bline=(-.55, -0.05))

    mi_results = {}
    pv_results = {}

    for _r, _mt, _inf in zip(regressors, mi_type, inference):

        _reg = [np.array(x[_r]) for x in rgd]

        if _mt == 'cc':
            _reg = [r.astype('float32') for r in _reg]
        elif _mt == 'cd':
            _reg = [r.astype('int32') for r in _reg]

        # _reg = [np.array(x[_r]) for x in rgd]

        ds_ephy = DatasetEphy(x=epd.copy(), y=_reg, roi=rois,
                              z=None, times=tp)
        wf = WfMi(mi_type=_mt, inference=_inf, kernel=np.hanning(20))
        mi, pval = wf.fit(ds_ephy, n_perm=1000, n_jobs=-1)
        mi['times'] = tp
        pval['times'] = tp

        mi_results[_r] = mi
        pv_results[_r] = pval

    ds_mi = xr.Dataset(mi_results)
    ds_pv = xr.Dataset(pv_results)

    return ds_mi, ds_pv


if __name__ == '__main__':
    monkeys = ['freddie', 'teddy']
    conditions = ['easy', 'hard']
    event = 'trig_off'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    # sectors = ['limbic striatum']
    overwrite = False

    regressors = ['Reward', 'q_rpe']
    # conditionals = ['Actions']
    mi_type = ['cd', 'cc']

    inference = ['ffx' for r in regressors]

    rej_files = []
    rej_files += ['1204', '1217', '1231', '0944',  # Bad sessions
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

                  '0220', '0223', '0271', '0273', '0278', '0280', '0284',
                  '0289', '0296', '0303', '0363', '0416', '0438', '0448',
                  '0521', '0618', '0656', '0691', '0693', '0698', '0705',
                  '0707', '0711', '0712', '0716', '0720', '0731']

    rej_files += ['0900', '1512', '1555', '1682',
                  '0291', '0368', '0743']

    for monkey in monkeys:
        for condition in conditions:
            dirs = get_dirs('local', 'lfp_causal')
            directory = dirs['pow'].format(monkey, condition)
            epo_dir = dirs['epo'].format(monkey, condition)
            regr_dir = dirs['reg'].format(monkey, condition)
            rec_info = op.join(dirs['ep_cnds'].format(monkey, condition),
                               'files_info.xlsx')

            all_files = []
            all_bline = []
            # all_conds = []
            all_rois = []
            # log_bads = []
            bad_epo = []
            reg_files = []
            for sect in sectors:
                fid = read_sector(rec_info, sect)
                fid = fid[fid['quality'] <= 3]
                # fid = fid[fid['neuron_type'] == 'TAN']


                for fs in fid['file']:
                    fname = op.join(epo_dir,
                                    '{0}_{1}_epo.fif'.format(fs, event))
                    rname = op.join(regr_dir, '{0}.xlsx'.format(fs))
                    if op.exists(fname) and fs not in rej_files:
                        all_files.append(fname)
                        # all_conds.append(rname)
                        reg_files.append(rname)
                        all_rois.append([sect])

                        be = get_ch_bad_epo(monkey, condition, fs,
                                            fname_info=rec_info)
                        bad_epo.append(be)

            ds_mi, ds_pv = compute_stats(all_files, reg_files, regressors,
                                         bad_epo, all_rois, mi_type,
                                         inference, times=(-1.5, 1.3),
                                         norm='cue_on')

            save_dir = op.join(dirs['st_prj'], monkey, condition, event,
                               'fbline_relchange', 'epochs')

            os.makedirs(save_dir, exist_ok=True)
            fname_mi = op.join(save_dir, 'mi_results.nc')
            fname_pv = op.join(save_dir, 'pv_results.nc')

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
