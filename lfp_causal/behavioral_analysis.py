import numpy as np
import os
import os.path as op
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
    p05 = np.percentile(rt, 5)
    p95 = np.percentile(rt, 95)
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
    p05 = np.percentile(md, 5)
    p95 = np.percentile(md, 95)
    fig, ax = plt.subplots(1, 1)
    ax.hist(md, bins, color='k')
    ax.axvline(p05, color='g', label='5th percentile: {0}'.format(p05))
    ax.axvline(p95, color='r', label='percentile: {0}'.format(p95))
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    directories = get_dirs('local', 'lfp_causal')

    monkeys = ['freddie']
    conditions = ['easy', 'hard']
    # conditions = ['hard']

    rej_files = ['1204', '1217', '1231', '0944',  # Bad sessions
                 '0845', '0847', '0939', '0946', '0963', '1036', '1231',
                 '1233', '1234', '1514', '1699',
                 '0940', '0944', '0964', '0967', '0969', '0970', '0971',
                 '0977', '0985', '1280']

    fnames_txt = []
    for mk in monkeys:
        for cd in conditions:
            info_dir = directories['inf'].format(mk, cd)
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

    reaction_times(fnames_txt, bins=300)
    movement_duration(fnames_txt, bins=300)
