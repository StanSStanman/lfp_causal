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

    ca_easy = []
    ca_hard = []
    for fn in fnames:
        xls = pd.read_excel(fn)

        ## Correct
        # correct = xls['Correct']
        # c_range = np.arange(1, len(correct) + 1)
        # c_csum = np.cumsum(correct)
        # c_prob = c_csum / c_range
        # if 'easy' in fn:
        #     ca_easy.append(pd.DataFrame(c_prob))
        # elif 'hard' in fn:
        #     ca_hard.append(pd.DataFrame(c_prob))

        ## Pcorr
        pcorr = xls['q_pcorr']
        if 'easy' in fn:
            ca_easy.append(pd.DataFrame(pcorr))
        elif 'hard' in fn:
            ca_hard.append(pd.DataFrame(pcorr))

        # plt.plot(c_prob)
        # ca.append(pd.DataFrame(correct))
        # plt.scatter(range(len(correct)), correct)
    plt.show()

    ca_easy = pd.concat(ca_easy, ignore_index=True, axis=1)
    ca_hard = pd.concat(ca_hard, ignore_index=True, axis=1)
    perc_easy = ca_easy.count(axis=1) / ca_easy.count(axis=1).max()
    perc_hard = ca_hard.count(axis=1) / ca_hard.count(axis=1).max()

    ## Correct
    # mean_easy = ca_easy.mean(axis=1, skipna=True)
    # sem_easy = ca_easy.sem(axis=1, skipna=True)
    # mean_hard = ca_hard.mean(axis=1, skipna=True)
    # sem_hard = ca_hard.sem(axis=1, skipna=True)

    ## Pcorr
    count_easy = ca_easy.count(axis=1)
    sem_easy = ca_easy.sem(axis=1, skipna=True)
    mean_easy = ca_easy.sum(axis=1, skipna=True) / count_easy
    count_hard = ca_hard.count(axis=1)
    sem_hard = ca_hard.sem(axis=1, skipna=True)
    mean_hard = ca_hard.sum(axis=1, skipna=True) / count_hard

    plt.rcParams["figure.figsize"] = (15, 10)
    plt.errorbar(range(len(mean_easy)), mean_easy, yerr=sem_easy,
                 color='navy', linewidth=2.5, linestyle='-',
                 label='P(C) easy')
    # plt.scatter(range(len(mean_easy)), mean_easy, color='navy', linewidth=1.5)
    plt.plot(perc_easy, color='slateblue', linewidth=2.5, linestyle='-',
             label='% trials per session easy')

    plt.errorbar(range(len(mean_hard)), mean_hard, yerr=sem_hard,
                 color='crimson', linewidth=2.5, linestyle='-',
                 label='P(C) hard')
    # plt.scatter(range(len(mean_hard)), mean_hard, color='crimson',
    #             linewidth=1.5)
    plt.plot(perc_hard, color='tomato', linewidth=2.5, linestyle='-',
             label='% trials per session hard')

    plt.title('Probability of correct responses across trials', fontsize=20)
    plt.xlabel('Trials', fontsize=16)
    plt.ylabel('P(correct)', fontsize=16)
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    directories = get_dirs('local', 'lfp_causal')

    monkeys = ['teddy']
    conditions = ['easy', 'hard']
    # conditions = ['easy']

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
