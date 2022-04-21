import numpy as np
import os
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from research.get_dirs import get_dirs
from lfp_causal.IO import read_txt, read_session
import seaborn as sns
sns.set_context('poster')


def reaction_times(fnames, monkeys, conditions, bins=30, division=None):

    rt = []
    ps = []
    monkeys = np.array(monkeys)
    conditions = np.array(conditions)
    for fn in fnames:
        txt_info = read_txt(fn)
        codes = np.array(txt_info['code'])
        bad_tr = np.logical_and(codes != '0', codes != '5700')
        react_time = np.array(txt_info['TR']).astype(float)
        react_time = np.delete(react_time, np.where(bad_tr))
        position = np.array(txt_info['N-conta']).astype(int)
        position = np.delete(position, np.where(bad_tr))
        # Removing exceding time trials
        react_time = np.delete(react_time, np.where(react_time == 1000))
        position = np.delete(position, np.where(react_time == 1000))
        # Cutting at 25 trials
        react_time = react_time[:25]
        position = position[:25]

        rt.append(react_time)
        ps.append(position)

    if division is None:
        rt = np.concatenate(rt)
        # rt = np.delete(rt, np.where(rt == 1000))
        rt /= 1000.
        p05 = np.percentile(rt, 5).round(3)
        p95 = np.percentile(rt, 95).round(3)
        fig, ax = plt.subplots(1, 1)
        ax.hist(rt, bins, color='k')
        ax.axvline(p05, color='g', label='5th percentile: {0}'.format(p05))
        ax.axvline(p95, color='r', label='95th percentile: {0}'.format(p95))
        plt.legend()
        plt.show()
    elif division is 'position':
        d = {'right': [], 'center': [], 'left': []}
        for _r, _p in zip(rt, ps):
            # _r = np.delete(_r, np.where(_r == 1000))
            # _p = np.delete(_p, np.where(_r == 1000))
            _r /= 1000.
            for i, v in enumerate(_p):
                if v == 2:
                    d['right'].append(_r[i])
                elif v == 3:
                    d['center'].append(_r[i])
                elif v == 4:
                    d['left'].append(_r[i])
        n_pos = np.arange(len(d.keys()))
        means = [np.array(d[k]).mean() for k in d.keys()]
        stds = [np.array(d[k]).std() for k in d.keys()]
        fig, ax = plt.subplots(1)
        p = ax.bar(n_pos, means, width=0.7, yerr=stds)
        ax.set_xticks(n_pos)
        ax.set_xticklabels(tuple(d.keys()))
        # plt.tight_layout()
        plt.ylim([0., .55])
        plt.show()
    # Divide data by monkeys and conditions
    elif division is 'mk_cd':
        # col = ['chartreuse', 'indigo']
        col = ['navy', 'crimson']
        d = {k: {} for k in np.unique(monkeys)}
        for k in d.keys():
            d[k] = {_k: [] for _k in np.unique(conditions)}
        for mk, cd, _r in zip(monkeys, conditions, rt):
            d[mk][cd].append(_r)
        fig, axs = plt.subplots(1, len(np.unique(monkeys)), figsize=(15, 6))
        fig.suptitle('Reaction times')
        for i, _mk in enumerate(np.unique(monkeys)):
            axs[i].set_title(_mk)
            for _i, _cd in enumerate(d[_mk].keys()):
                _rt = np.concatenate(d[_mk][_cd])
                _rt /= 1000.
                mean = _rt.mean()
                axs[i].hist(_rt, bins, color=col[_i], alpha=.6)
                axs[i].axvline(mean, color=col[_i], label=_cd)
            axs[i].legend()
        # plt.legend()
        plt.tight_layout()
        plt.show()
    # Divide data by monkeys
    elif division is 'mk':
        # col = ['chartreuse', 'indigo']
        col = ['navy', 'crimson']
        # col = ['chartreuse', 'crimson']
        d = {k: [] for k in np.unique(monkeys)}
        # for k in d.keys():
        #     d[k] = {_k: [] for _k in np.unique(conditions)}
        for mk, _r in zip(monkeys, rt):
            d[mk].append(_r)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle('Reaction times')
        for i, _mk in enumerate(np.unique(monkeys)):
            _rt = np.concatenate(d[_mk])
            _rt /= 1000.
            mean = _rt.mean()
            ax.hist(_rt, bins, color=col[i], alpha=.6)
            ax.axvline(mean, color=col[i],
                       label=_mk + ' {0}'.format(str(mean.round(3))))
        ax.legend()
        # plt.tight_layout()
        plt.show()

    return


def movement_duration(fnames, monkeys, conditions, bins=30, division=None):

    md = []
    ps = []
    monkeys = np.array(monkeys)
    conditions = np.array(conditions)
    for fn in fnames:
        txt_info = read_txt(fn)
        codes = np.array(txt_info['code'])
        bad_tr = np.logical_and(codes != '0', codes != '5700')
        mov_dur = np.array(txt_info['TM']).astype(float)
        mov_dur = np.delete(mov_dur, np.where(bad_tr))
        position = np.array(txt_info['N-conta']).astype(int)
        position = np.delete(position, np.where(bad_tr))
        # Removing exceding time trials
        mov_dur = np.delete(mov_dur, np.where(mov_dur == 1000))
        position = np.delete(position, np.where(mov_dur == 1000))
        # Cutting at 25 trials
        mov_dur = mov_dur[:25]
        position = position[:25]

        md.append(mov_dur)
        ps.append(position)

    if division is None:
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
    elif division is 'position':
        d = {'right': [], 'center': [], 'left': []}
        for _m, _p in zip(md, ps):
            # _m = np.delete(_m, np.where(_m == 1000))
            # _p = np.delete(_p, np.where(_m == 1000))
            _m /= 1000.
            for i, v in enumerate(_p):
                if v == 2:
                    d['right'].append(_m[i])
                elif v == 3:
                    d['center'].append(_m[i])
                elif v == 4:
                    d['left'].append(_m[i])
        n_pos = np.arange(len(d.keys()))
        means = [np.array(d[k]).mean() for k in d.keys()]
        stds = [np.array(d[k]).std() for k in d.keys()]
        fig, ax = plt.subplots(1)
        p = ax.bar(n_pos, means, width=0.7, yerr=stds)
        ax.set_xticks(n_pos)
        ax.set_xticklabels(tuple(d.keys()))
        # plt.tight_layout()
        plt.ylim([0., .55])
        plt.show()
    elif division is 'mk_cd':
        # col = ['chartreuse', 'indigo']
        col = ['navy', 'crimson']
        d = {k: {} for k in np.unique(monkeys)}
        for k in d.keys():
            d[k] = {_k: [] for _k in np.unique(conditions)}
        for mk, cd, _m in zip(monkeys, conditions, md):
            d[mk][cd].append(_m)
        fig, axs = plt.subplots(1, len(np.unique(monkeys)), figsize=(15, 6))
        fig.suptitle('Movement durations')
        for i, _mk in enumerate(np.unique(monkeys)):
            axs[i].set_title(_mk)
            for _i, _cd in enumerate(d[_mk].keys()):
                _md = np.concatenate(d[_mk][_cd])
                _md /= 1000.
                mean = _md.mean()
                axs[i].hist(_md, bins, color=col[_i], alpha=.6)
                axs[i].axvline(mean, color=col[_i], label=_cd)
            axs[i].legend()
        # plt.legend()
        plt.tight_layout()
        plt.show()
    elif division is 'mk':
        # col = ['chartreuse', 'indigo']
        col = ['navy', 'crimson']
        # col = ['chartreuse', 'crimson']
        d = {k: [] for k in np.unique(monkeys)}
        # for k in d.keys():
        #     d[k] = {_k: [] for _k in np.unique(conditions)}
        for mk, _r in zip(monkeys, md):
            d[mk].append(_r)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle('Movement durations')
        for i, _mk in enumerate(np.unique(monkeys)):
            _md = np.concatenate(d[_mk])
            _md /= 1000.
            mean = _md.mean()
            ax.hist(_md, bins, color=col[i], alpha=.6)
            ax.axvline(mean, color=col[i],
                       label=_mk + ' {0}'.format(str(mean.round(3))))
        ax.legend()
        # plt.tight_layout()
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
        pcorr = xls['Correct']
        # Append prob. values cut at 25 trials
        if 'easy' in fn:
            ca_easy.append(pd.DataFrame(pcorr[:25]))
        elif 'hard' in fn:
            ca_hard.append(pd.DataFrame(pcorr[:25]))

        # plt.plot(c_prob)
        # ca.append(pd.DataFrame(correct))
        # plt.scatter(range(len(correct)), correct)
    # plt.show()

    ca_easy = pd.concat(ca_easy, ignore_index=True, axis=1)
    ca_hard = pd.concat(ca_hard, ignore_index=True, axis=1)
    perc_easy = ca_easy.count(axis=1) / ca_easy.count(axis=1).max()
    perc_hard = ca_hard.count(axis=1) / ca_hard.count(axis=1).max()

    ## Correct
    mean_easy = ca_easy.mean(axis=1, skipna=True)
    sem_easy = ca_easy.sem(axis=1, skipna=True)
    mean_hard = ca_hard.mean(axis=1, skipna=True)
    sem_hard = ca_hard.sem(axis=1, skipna=True)

    ## Pcorr
    # count_easy = ca_easy.count(axis=1)
    # sem_easy = ca_easy.sem(axis=1, skipna=True)
    # mean_easy = ca_easy.sum(axis=1, skipna=True) / count_easy
    # count_hard = ca_hard.count(axis=1)
    # sem_hard = ca_hard.sem(axis=1, skipna=True)
    # mean_hard = ca_hard.sum(axis=1, skipna=True) / count_hard

    plt.rcParams["figure.figsize"] = (15, 10)
    # Plot prob. easy with errorbar
    plt.errorbar(range(len(mean_easy)), mean_easy, yerr=sem_easy,
                 color='navy', linewidth=2.5, linestyle='-',
                 label='P(C) easy')
    # plt.scatter(range(len(mean_easy)), mean_easy, color='navy', linewidth=1.5)
    # Plot perc. of trials in easy cond.
    # plt.plot(perc_easy, color='slateblue', linewidth=2.5, linestyle='-',
    #          label='% trials per session easy')
    # Plot prob. hard with errorbar
    plt.errorbar(range(len(mean_hard)), mean_hard, yerr=sem_hard,
                 color='crimson', linewidth=2.5, linestyle='-',
                 label='P(C) hard')
    # plt.scatter(range(len(mean_hard)), mean_hard, color='crimson',
    #             linewidth=1.5)
    # Plot perc. of trials in hard cond.
    # plt.plot(perc_hard, color='tomato', linewidth=2.5, linestyle='-',
    #          label='% trials per session hard')

    plt.title('Probability of correct responses across trials', fontsize=20)
    plt.xlabel('Trials', fontsize=16)
    plt.ylabel('P(correct)', fontsize=16)
    plt.legend()
    plt.show()

    return


def prob_correct_answer_easy_hard(fnames, bads=None):

    ca_fr = []
    ca_te = []
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
        pcorr = xls['Correct']
        # pcorr = xls['Reward']
        # Append prob. values cut at 25 trials
        # ca.append(pd.DataFrame(pcorr[:25]))
        if 'freddie' in fn:
            ca_fr.append(pd.DataFrame(pcorr[:25]))
        elif 'teddy' in fn:
            ca_te.append(pd.DataFrame(pcorr[:25]))

        # plt.plot(c_prob)
        # ca.append(pd.DataFrame(correct))
        # plt.scatter(range(len(correct)), correct)
    # plt.show()

    ca_fr = pd.concat(ca_fr, ignore_index=True, axis=1)
    ca_te = pd.concat(ca_te, ignore_index=True, axis=1)
    perc_fr = ca_fr.count(axis=1) / ca_fr.count(axis=1).max()
    perc_te = ca_te.count(axis=1) / ca_te.count(axis=1).max()

    ## Correct
    mean_fr = ca_fr.mean(axis=1, skipna=True)
    sem_fr = ca_fr.sem(axis=1, skipna=True)
    mean_te = ca_te.mean(axis=1, skipna=True)
    sem_te = ca_te.sem(axis=1, skipna=True)

    ## Pcorr
    # count_easy = ca_easy.count(axis=1)
    # sem_easy = ca_easy.sem(axis=1, skipna=True)
    # mean_easy = ca_easy.sum(axis=1, skipna=True) / count_easy
    # count_hard = ca_hard.count(axis=1)
    # sem_hard = ca_hard.sem(axis=1, skipna=True)
    # mean_hard = ca_hard.sum(axis=1, skipna=True) / count_hard

    plt.rcParams["figure.figsize"] = (15, 10)
    # Plot prob. easy with errorbar
    plt.errorbar(range(len(mean_fr)), mean_fr, yerr=sem_fr,
                 color='navy', linewidth=4, linestyle='-',
                 label='P(C) Monkey F')
    # plt.scatter(range(len(mean_easy)), mean_easy, color='navy', linewidth=1.5)
    # Plot perc. of trials in easy cond.
    # plt.plot(perc_easy, color='slateblue', linewidth=2.5, linestyle='-',
    #          label='% trials per session easy')
    # Plot prob. hard with errorbar
    plt.errorbar(range(len(mean_te)), mean_te, yerr=sem_te,
                 color='crimson', linewidth=4, linestyle='-',
                 label='P(C) Monkey T')
    # plt.scatter(range(len(mean_hard)), mean_hard, color='crimson',
    #             linewidth=1.5)
    # Plot perc. of trials in hard cond.
    # plt.plot(perc_hard, color='tomato', linewidth=2.5, linestyle='-',
    #          label='% trials per session hard')

    plt.title('Probability of correct responses across trials', fontsize=28)
    plt.xlabel('Trials', fontsize=25)
    plt.ylabel('P(correct)', fontsize=25)
    plt.legend(loc='upper left')
    plt.show()

    return


def prob_correct_answer_ft_eh(fnames, bads=None):

    fr_ea = []
    fr_ha = []
    te_ea = []
    te_ha = []
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
        pcorr = xls['Correct']
        # pcorr = xls['Reward']
        # Append prob. values cut at 25 trials
        # ca.append(pd.DataFrame(pcorr[:25]))
        if 'freddie' in fn:
            if 'easy' in fn:
                fr_ea.append(pd.DataFrame(pcorr[:25]))
            elif 'hard' in fn:
                fr_ha.append(pd.DataFrame(pcorr[:25]))
        elif 'teddy' in fn:
            if 'easy' in fn:
                te_ea.append(pd.DataFrame(pcorr[:25]))
            elif 'hard' in fn:
                te_ha.append(pd.DataFrame(pcorr[:25]))

        # plt.plot(c_prob)
        # ca.append(pd.DataFrame(correct))
        # plt.scatter(range(len(correct)), correct)
    # plt.show()

    fr_ea = pd.concat(fr_ea, ignore_index=True, axis=1)
    fr_ha = pd.concat(fr_ha, ignore_index=True, axis=1)
    te_ea = pd.concat(te_ea, ignore_index=True, axis=1)
    te_ha = pd.concat(te_ha, ignore_index=True, axis=1)
    perc_frea = fr_ea.count(axis=1) / fr_ea.count(axis=1).max()
    perc_frha = fr_ha.count(axis=1) / fr_ha.count(axis=1).max()
    perc_teea = te_ea.count(axis=1) / te_ea.count(axis=1).max()
    perc_teha = te_ha.count(axis=1) / te_ha.count(axis=1).max()

    ## Correct
    mean_frea = fr_ea.mean(axis=1, skipna=True)
    sem_frea = fr_ea.sem(axis=1, skipna=True)
    mean_frha = fr_ha.mean(axis=1, skipna=True)
    sem_frha = fr_ha.sem(axis=1, skipna=True)
    mean_teea = te_ea.mean(axis=1, skipna=True)
    sem_teea = te_ea.sem(axis=1, skipna=True)
    mean_teha = te_ha.mean(axis=1, skipna=True)
    sem_teha = te_ha.sem(axis=1, skipna=True)

    plt.rcParams["figure.figsize"] = (13, 9)
    labels = ['Monkey F, easy', 'Monkey F, hard',
              'Monkey T, easy', 'Monkey T, hard']
    colors = ['maroon', 'indianred',
              'navy', 'cornflowerblue']

    for pl, lb, cl in zip([(mean_frea, sem_frea), (mean_frha, sem_frha),
                       (mean_teea, sem_teea), (mean_teha, sem_teha)],
                      labels, colors):
        plt.fill_between(range(len(pl[1])), pl[0]-pl[1], pl[0]+pl[1],
                         color=cl, alpha=.25)
        plt.plot(range(len(pl[0])), pl[0], color=cl, linewidth=3,
                 linestyle='-', marker='d', ms=3, label=lb)

    #plt.title('Probability of correct responses across trials', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Trials', fontsize=22)
    plt.ylabel('P(correct)', fontsize=22)
    plt.legend(loc='lower right', fontsize=22)
    plt.savefig('/home/jerry/Immagini/new_imgs/learning_curve_fr_te_ea_ha.svg')
    plt.savefig('/home/jerry/Immagini/new_imgs/pngs/'
                'learning_curve_fr_te_ea_ha.png')
    plt.show()

    return


def prob_wstay_lshift(fnames_xls, n_tr=7):
    wst = 0
    lsh = 0
    trl = 0
    for fn in fnames_xls:
        xls = pd.read_excel(fn)
        rewards = xls['Reward']
        actions = xls['Actions']
        for t in range(n_tr):
            _r = rewards[t]
            _a = actions[t]
            if _r == 1 and _a == actions[t + 1]:
                wst += 1
            elif _r == 0 and _a != actions[t+1]:
                lsh += 1
        trl += n_tr

    return wst, lsh, trl


def plot_wst_lsh(fnames_xls):
    d = {'fr_ea': [],
         'fr_ha': [],
         'te_ea': [],
         'te_ha': []}

    for fn in fnames_xls:
        if 'freddie' in fn and 'easy' in fn:
            d['fr_ea'].append(fn)
        elif 'freddie' in fn and 'hard' in fn:
            d['fr_ha'].append(fn)
        elif 'teddy' in fn and 'easy' in fn:
            d['te_ea'].append(fn)
        elif 'teddy' in fn and 'hard' in fn:
            d['te_ha'].append(fn)

    wst, lsh = [], []
    for k in d.keys():
        _wst, _lsh, _tn = prob_wstay_lshift(d[k])
        wst.append(_wst / _tn)
        lsh.append(_lsh / _tn)

    labels = ['freddie easy', 'freddie hard', 'teddy easy', 'teddy hard']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ln_wst = ax.bar(x - width/2, wst, width, label='win-stay')
    ln_lsh = ax.bar(x + width/2, lsh, width, label='lose-shift')

    ax.set_ylabel('Occurrences')
    ax.set_title('Number of win-stay and lose-shift')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # ax.bar_label(ln_wst, padding=3)
    # ax.bar_label(ln_lsh, padding=3)

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    directories = get_dirs('local', 'lfp_causal')

    # monkeys = ['freddie']
    # monkeys = ['teddy']
    monkeys = ['freddie', 'teddy']
    conditions = ['easy', 'hard']

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
    mks = []
    cds = []
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
                            mks.append(mk)
                            cds.append(cd)
                    except:
                        continue

            for f in os.listdir(regr_dir):
                if 'act' not in f:
                    session = f.replace('.xlsx', '')
                    info = read_session(rec_info, session)
                    if info['quality'].values <= 3 and \
                            session not in rej_files:
                        fnames_xls.append(op.join(regr_dir, f))

    # reaction_times(fnames_txt, mks, cds, bins=100, division='mk')
    # movement_duration(fnames_txt, mks, cds, bins=100, division='mk')
    # prob_correct_answer(fnames_xls)
    # prob_correct_answer_easy_hard(fnames_xls)
    prob_correct_answer_ft_eh(fnames_xls)
    # plot_wst_lsh(fnames_xls)
