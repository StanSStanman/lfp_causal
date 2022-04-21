# TO SUPPRESS

import mne
import numpy as np
import scipy as sp
from scipy.io import loadmat
from collections import OrderedDict
from openpyxl import load_workbook
import pandas as pd
import os
import json
import matplotlib.pyplot as plt


directory = 'D:\\Databases\\db_lfp\\meg_causal\\'
xls_fname = 'dataset_StriPAN_proba-{0}.xlsx'
subject = ['freddie']
condition = ['easy']
session = ['fneu1218']
rawmat_dir = directory + '{0}\\{1}\\raw_matlab\\' #{2}.mat' #.format(subject, condition, session)
info_dir = directory + '{0}\\{1}\\infos\\' #info_{2}.json' #.format(subject, condition, session)
neu_dir = directory + '{0}\\{1}\\neu_data\\' #{2}.npz' #.format(subject, condition, session)
beh_dir = directory + '{0}\\{1}\\beh_data\\' #{2}.npz' #.format(subject, condition, session)
raw_dir = directory + '{0}\\{1}\\raw\\'
# prep_dir = directory + '{0}\\prep\\{1}\\{0}_outcome-epo.fif'

def create_folders(directory, subject, condition):
    '''
    :param directory: str, main directory of the lfp database
    :param subject: list of str, name of the subjects
    :param condition: list of str, task condition

    :return: create the folders to store the data
    '''
    folders = ['infos', 'beh_data', 'neu_data', 'prep', 'epochs', 'raw']
    for sub in subject:
        for cond in condition:
            for f in folders:
                if not os.path.exists(directory + '{0}\\{1}\\{2}'.format(sub, cond, f)):
                    os.makedirs(directory + '{0}\\{1}\\{2}'.format(sub, cond, f))


def save_info(directory, xls, subject, condition, info_dir, rawmat_dir):
    '''
    :param directory: str, main directory of the lfp database
    :param xls: str, name of the xls file cointaining the information
    :param subject: list of str, names of the subjects to iterate
    :param condition: list of str, conditions of the task to iterate
    :param info_dir: str, name of the directory in which the infos will be saved
    :param rawmat_dir: str, name of the directory containing the raw mat files

    :return: saves all the info in different json files (dict),
             save a file containing the name of the present and missing mat file (list of lists)
             save a file with the info labels (list)
    '''
    for sub in subject:
        for cond in condition:
            ld_xls = pd.read_excel(directory + xls.format(cond), sheet_name=sub)
            ld_xls = ld_xls.rename(columns={'Best block-target': 'block_target'})
            items = list(ld_xls.columns.values.astype(str))
            date = items.pop(1)
            items_form = [str, int, int, str, int, str, int, int, str, str, str]
            for i, fi in zip(items, items_form):
                ld_xls[i] = ld_xls[i].apply(fi)
            ld_xls[date] = ld_xls[date].dt.strftime('%d/%m/%Y')
            items.insert(1, date)
            # ld_xls['file'] = ld_xls['file'].apply(str)
            for f, n in zip(ld_xls['file'], range(ld_xls.shape[0])):
                if len(f) < 4:
                    for l in range(4 - len(f)):
                        f = '0' + f
                    ld_xls['file'][n] = f
            for s in ld_xls.iterrows():
                s[1].to_json(path_or_buf=info_dir.format(sub, cond) + 'info_{0}.json'.format(s[1]['file']),
                             orient='index', force_ascii=False)
            files = ld_xls['file'].tolist()
            pres_f, abs_f = [], []
            for fn in files:
                if os.path.isfile(rawmat_dir.format(sub, cond) + 'fneu{0}.mat'.format(fn)):
                    pres_f.append('fneu'+fn)
                else: abs_f.append('fneu'+fn)
            print(len(abs_f), 'files not found for {0}, condition {1}:'.format(sub, cond), '\n', abs_f)
            np.save(info_dir.format(sub, cond) + 'files_info', [pres_f, abs_f])
            np.save(info_dir.format(sub, cond) + 'info_args', items)


def create_rawfiles(subject, condition, session, rawmat_dir):
    data_dict = OrderedDict()
    data_labels = ['lfp', 'mua', 'time', 'trigger_time', 'action_time',
                   'contact_time', 'action', 'outcome']

    for sub in subject:
        for cond in condition:
            for ses in session:
                matfile = loadmat(rawmat_dir.format(sub, cond) + '{0}'.format(ses))
                mat_data = matfile['data']

                for d, l in zip(mat_data[0][0], data_labels):
                    data_dict[l] = d

                time = data_dict['time'].astype(float)
                neu_data = np.vstack((data_dict['lfp'].astype(float), data_dict['mua'].astype(float)))#, data_dict['time']))

                ch_types = ['seeg', 'seeg']
                ch_names = ['lfp', 'mua']
                sfreq = 16667
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                raw = mne.io.RawArray(neu_data, info)

                raw_lfp = raw.copy().pick_channels(['lfp'])
                raw_mua = raw.copy().pick_channels(['mua'])

                act_times = data_dict['action_time'].copy() * 1000
                act_times = np.around(act_times, decimals=0).astype(int)
                outc_times = data_dict['contact_time'].copy() * 1000
                outc_times = np.around(outc_times, decimals=0).astype(int)

                action_events = np.hstack((act_times, np.zeros((len(act_times), 1), dtype=int), data_dict['action'].astype(int)))
                outcome_events = np.hstack((outc_times, np.zeros((len(outc_times), 1), dtype=int), data_dict['outcome'].astype(int)))

                epo_lfp_act = mne.Epochs(raw_lfp, action_events)
                epo_lfp_outc = mne.Epochs(raw_lfp, outcome_events)
                epo_mua_act = mne.Epochs(raw_mua, action_events)
                epo_mua_outc = mne.Epochs(raw_mua, outcome_events)

                freqs = np.arange(5.0, 60.0, 3.0)
                n_cycles = freqs / 5
                tfr = mne.time_frequency.tfr_morlet(epo_lfp_act, freqs, n_cycles, return_itc=False, average=True)
                tfr.plot([0], baseline=(0.3, 0.5), mode='zlogratio', tmin=-0.15, tmax=0.45, vmin=-3, vmax=3)

                scalings = {'seeg':2}
                raw.plot(n_channels = 2, scalings=scalings, show=True, block=True)

                # from mne.time_frequency.tfr import cwt
                # mua = neu_data[1, :]
                # mua = mua.astype(float)
                # mua = np.expand_dims(mua, 0)
                # time = mat_data[0][0][2].astype(float)
                # wave = cwt(mua, time, use_fft=False, decim=1667)

    print('ciao')


if __name__ == '__main__':
    # save_info(directory, xls_fname, condition, subject, info_dir, rawmat_dir)
    create_rawfiles(subject, condition, session, rawmat_dir)