import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.io import loadmat
from lfp_causal.directories import info_dir, rawmat_dir

def info_labels(subj, cond):
    ''' Read the labels in the matlab file

    :param subj: str, name of the subject
    :param cond: str, name of the condition
    :return: list, list of labels
    '''
    return np.load(info_dir.format(subj, cond) + 'info_args.npy')

def files_info(subj, cond):
    ''' Read the info about which matlab file exists for a subject/condition

    :param subj: str, name of the subject
    :param cond: str, name of the condition
    :return: list of list, a list in the form [['existing_files'], ['lacking_files']]
    '''
    return np.load(info_dir.format(subj, cond) + 'files_info.npy')

def trial_info(subj, cond, fname):
    ''' Read the trial related information

    :param subj: str, name of the subject
    :param cond: str, name of the condition
    :param fname: str, name of the session
    :return: pandas Series, with a list of name and the relative value
    '''
    return pd.read_json(info_dir.format(subj, cond) + 'info_{0}.json'.format(fname), orient='index', typ='series')

def read_matfile(subject, condition, session):
    ''' Read the matlab file in the 'rawmat_dir' folder

    :param subject: str, name of the subject
    :param condition: str, name of the condition
    :param session: str, name of the session
    :return: dict, a dictionary containing the informations about:
             ['lfp', 'mua', 'time', 'trigger_time', 'action_time', 'contact_time', 'action', 'outcome']
    '''
    data_dict = OrderedDict()
    data_labels = ['lfp', 'mua', 'time', 'trigger_time', 'action_time', 'contact_time', 'action', 'outcome']

    if 'fneu' not in session:
        session = 'fneu' + session

    matfile = loadmat(rawmat_dir.format(subject, condition) + '{0}'.format(session))
    mat_data = matfile['data']

    for d, l in zip(mat_data[0][0], data_labels):
        data_dict[l] = d

    return data_dict

def write_ch_info(fname_in, fname_out):
    labels = ['monkey', 'file', 'n_neur', 'type', 'record', 'sector', 'test', 'best_target', 'quality']
    data_dic = {k: [] for k in labels}
    lines = open(fname_in, encoding='utf-8').read().split('\n')
    if 'monkey' in lines[0]: lines.remove(lines[0])
    for l in lines:
        l = l.split('\t')
        l = list(filter(lambda x: x != '', l))
        for v in l:
            if 'channel' in v:
                l.remove(v)
        if len(l) > 0:
            if len(l[1]) == 3:
                l[1] = '0' + l[1]
        if len(l) == 8: l.append('5')
        for k, i in zip(labels, l):
            data_dic[k].append(i)
    df = pd.DataFrame.from_dict(data_dic)
    df.to_excel(fname_out, sheet_name=df['monkey'][0], index=False)

def read_ch_info(fname, subject, quality):
    df = pd.read_excel(fname, sheet_name=subject, dtype={'file': str})
    df = df.loc[df['quality'] <= quality]
    return df

# if __name__ == '__main__':
#     write_ch_info('D:\\Databases\\db_lfp\\lfp_causal\\LFP analyses-Freddie.txt')