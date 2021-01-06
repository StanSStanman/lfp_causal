import numpy as np
import pandas as pd
import mne
from ast import literal_eval
from lfp_causal.IO import read_bad_epochs


def compute_bad_epo(epoch, xls_bad):
    if isinstance(epoch, str):
        bad_log = mne.read_epochs(epoch, preload=False).drop_log
    else:
        bad_log = epoch.drop_log

    _b = []
    for i, d in enumerate(bad_log):
        if d != ():
            _b.append(i)

    if xls_bad == None:
        xls_bad = []

    join_bads = list(np.unique(xls_bad + _b))

    return join_bads


def get_log_bad_epo(epoch):
    if isinstance(epoch, str):
        dl = mne.read_epochs(epoch, preload=False).drop_log
    elif isinstance(epoch, mne.Epochs):
        dl = epoch.drop_log
    _b = []
    for i, d in enumerate(dl):
        if d != ():
            _b.append(i)

    return _b


def get_ch_bad_epo(monkey, condition, session, fname_info=None):
    if fname_info is None:
        fname_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
                        '{0}/{1}/files_info.xlsx'.format(monkey, condition)
    assert isinstance(fname_info, str)

    xls = pd.read_excel(fname_info,
                        dtype={'file': str, 'good_channel': int,
                               'bad_LFP1': list, 'bad_LFP2': list})
    good_ch = xls['good_channel'][xls['file'] == session].values[0]

    if good_ch == 0:
        good_ch = 1
    if good_ch == 1:
        _bi = xls['bad_LFP1'][xls['file'] == session].values[0]
    elif good_ch == 2:
        _bi = xls['bad_LFP2'][xls['file'] == session].values[0]

    b = literal_eval(_bi)
    # if _bi.values[0] == 'None':
    #     b = None
    # else:
    #     b = list(map(int, _bi.values[0].split(',')))

    return b


# def write_bad_epo(monkey, condition, session):
