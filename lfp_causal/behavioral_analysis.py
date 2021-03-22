import numpy as np
import os
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from research.get_dirs import get_dirs
from lfp_causal.IO import read_txt, read_session

def reaction_times(fnames, bins=30, bads=None):

    rt = []
    for fn in fnames:
        txt_info = read_txt(fn)
        codes = np.array(txt_info['code'])
        bad_tr = np.logical_and(codes != '0', codes != '5700')
        react_time = np.array(txt_info['TR']).astype(float)
        react_time = np.delete(react_time, np.where(bad_tr))
        rt.append(react_time)
    rt = np.concatenate(rt)
    rt = np.delete(rt, np.where(rt == 1000))
    rt /= 1000.
    p05 = np.percentile(rt, 5).round(3)
    p95 = np.percentile(rt, 95).round(3)
    fig, ax = plt.subplots(1, 1)
    ax.hist(rt, bins, color='k')
    ax.axvline(p05, color='g', label='5th percentile: {0}'.format(p05))
    ax.axvline(p95, color='r', label='95th percentile: {0}'.format(p95))
    plt.legend()
    plt.show()

    return


def movement_duration(fnames, bins=30, bads=None):

    md = []
    for fn in fnames:
        txt_info = read_txt(fn)
        codes = np.array(txt_info['code'])
        bad_tr = np.logical_and(codes != '0', codes != '5700')
        mov_dur = np.array(txt_info['TM']).astype(float)
        mov_dur = np.delete(mov_dur, np.where(bad_tr))
        md.append(mov_dur)
    md = np.concatenate(md)
    # md = np.delete(md, np.where(md == 0))
    # md = np.delete(md, np.where(md == 1000))
    md /= 1000.
    p05 = np.percentile(md, 5).round(3)
    p95 = np.percentile(md, 95).round(3)
    fig, ax = plt.subplots(1, 1)
    ax.hist(md, bins, color='k')
    ax.axvline(p05, color='g', label='5th percentile: {0}'.format(p05))
    ax.axvline(p95, color='r', label='95th percentile: {0}'.format(p95))
    plt.legend()
    plt.show()

    return


def prob_correct_answer(fnames, bads=None):

    ca = []
    for fn in fnames:
        xls = pd.read_excel(fn)
        correct = xls['Correct']
        c_range = np.arange(1, len(correct) + 1)
        c_csum = np.cumsum(correct)
        c_prob = c_csum / c_range

        ca.append(pd.DataFrame(c_prob))
        plt.plot(c_prob)
        # ca.append(pd.DataFrame(correct))
        # plt.scatter(range(len(correct)), correct)
    plt.show()

    ca = pd.concat(ca, ignore_index=True, axis=1)
    mean = ca.mean(axis=1, skipna=True)
    sem = ca.sem(axis=1, skipna=True)

    plt.plot(mean, color='b', linewidth=2.5, linestyle='-')
    plt.scatter(range(len(mean)), mean, color='b', linewidth=1.5)
    plt.title('Probability of correct responses across trials')
    plt.xlabel('Trials')
    plt.ylabel('P(correct)')
    plt.show()


if __name__ == '__main__':
    directories = get_dirs('local', 'lfp_causal')

    monkeys = ['freddie']
    conditions = ['easy']
    # conditions = ['easy']

    rej_files = []
    rej_files += ['1204', '1217', '1231', '0944',  # Bad sessions
                  '0845', '0847', '0939', '0946', '0963', '1036', '1231',
                  '1233', '1234', '1514', '1699',
                  '0940', '0944', '0964', '0967', '0969', '0970', '0971',
                  '0977', '0985', '1037', '1280']
    rej_files += ['0210', '0226', '0227', '0230', '0362', '0365', '0393',
                  '0415', '0447', '0449', '0450', '0456', '0541', '0573',
                  '0622', '0628', '0631', '0643', '0653', '0660', '0706',
                  '0713', '0726', '0732',
                  '0296', '0363', '0416', '0438', '0448', '0521', '0705',
                  '0707', '0712', '0731']

    fnames_txt = []
    fnames_xls = []
    for mk in monkeys:
        for cd in conditions:
            info_dir = directories['inf'].format(mk, cd)
            regr_dir = directories['reg'].format(mk, cd)
            rec_info = op.join(directories['ep_cnds'].format(mk, cd),
                               'files_info.xlsx')
            epo_dir = directories['epo'].format(mk, cd)


            for f in os.listdir(info_dir):
                if f.endswith('.txt'):
                    if mk == 'freddie':
                        session = f.replace('fneu', '')
                    elif mk == 'teddy':
                        session = f.replace('tneu', '')
                    session = session.replace('.txt', '')
                    try:
                        info = read_session(rec_info, session)
                        if info['quality'].values <= 3 and \
                                session not in rej_files:
                            fnames_txt.append(op.join(info_dir, f))
                    except:
                        continue

            for f in os.listdir(regr_dir):
                if 'act' not in f:
                    session = f.replace('.xlsx', '')
                    info = read_session(rec_info, session)
                    if info['quality'].values <= 3 and \
                            session not in rej_files:
                        fnames_xls.append(op.join(regr_dir, f))

    # reaction_times(fnames_txt, bins=300)
    # movement_duration(fnames_txt, bins=300)
    prob_correct_answer(fnames_xls)
