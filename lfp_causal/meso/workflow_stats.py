import numpy as np
from lfp_causal.compute_stats import prepare_data


def data_flow(data, regr, rois, cond=None, mi_type='cc'):
    u_rois = np.unique(rois)
    _data = {k: [] for k in u_rois}
    _regr = {k: [] for k in u_rois}
    for roi, dt, rg in zip(rois, data, regr):
        _data[roi[0]].append(dt)
        _regr[roi[0]].append(rg)
    del data
    del regr

    if cond is not None:
        _cond = {k: [] for k in u_rois}
        for roi, cn in zip(rois, cond):
            _cond[roi[0]].append(cn)
        del cond

    for k in u_rois:
        _data[k] = np.concatenate(_data[k], axis=0).transpose(1, 2, 3, 0)
        _regr[k] = np.concatenate(_regr[k], axis=0).transpose(1, 2, 3, 0)

    if cond is not None:
        for k in u_rois:
            _cond[k] = np.concatenate(_cond[k], axis=0).transpose(1, 2, 3, 0)
    else:
        _cond = None

    return _data, _regr, _cond, u_rois


# def mi_flow():
# def stat_flow():