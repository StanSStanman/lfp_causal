import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
from research.get_dirs import get_dirs
from lfp_causal.IO import read_txt

def reaction_times(fnames, bins=30, bads=None):

    rt = []
    for fn in fnames:
        txt_info = read_txt(fn)
        react_time = np.array(txt_info['TR']).astype(float)
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
        mov_dur = np.array(txt_info['TM']).astype(float)
        md.append(mov_dur)
    md = np.concatenate(md)
    md = np.delete(md, np.where(md == 0))
    md = np.delete(md, np.where(md == 1000))
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

    fnames_txt = []
    for mk in monkeys:
        for cd in conditions:
            info_dir = directories['inf'].format(mk, cd)
            rec_info = op.join(directories['ep_cnds'].format(mk, cd),
                               'files_info.xlsx')
            epo_dir = directories['epo'].format(mk, cd)

            for f in os.listdir(info_dir):
                if f.endswith('.txt'):
                    fnames_txt.append(op.join(info_dir, f))
    reaction_times(fnames_txt, bins=500)
    movement_duration(fnames_txt, bins=500)
