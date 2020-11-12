import neo
import mne


def read_smr(fname):
    reader = neo.io.Spike2IO(filename=fname, try_signal_grouping=False)
    data = reader.read(lazy=False)[0]
    return data


def read_txt(fname):
    with open(fname, 'rb') as file:
        cont = file.read()

    cont = cont.decode('windows-1252')
    cont = cont.split('\n')
    for i, s in enumerate(cont):
        cont[i] = cont[i].replace('\r', '')

    keys = cont.pop(0).split('\t')
    data = {k: [] for k in keys}

    for s in cont:
        for v, k in zip(s.split('\t'), keys):
            data[k].append(v)

    return data


def read_sfreq(fname):
    raw = mne.io.read_raw_fif(fname)
    sfreq = raw.info['sfreq']

    return sfreq


def read_bad_epochs(monkey, condition, session):
    import numpy as np
    import pandas as pd
    import os.path as op
    csv = pd.read_csv(op.join('/media/jerry/TOSHIBA EXT/data/db_behaviour/',
                              'lfp_causal/{0}/{1}/'.format(monkey, condition),
                              'lfp_bad_trials.csv'), dtype=str)
    idx = np.where(csv['session'] == session)
    bads = csv.loc[idx]

    return bads


def read_xls(fname):
    import pandas as pd

    xls = pd.read_excel(fname, dtype='str')

    return xls


def read_sector(fname, sector):
    xls = read_xls(fname)
    sect_fid = xls[['file', 'sector']][xls['sector'] == sector]
    sect_fid = sect_fid.astype(str)

    return sect_fid
