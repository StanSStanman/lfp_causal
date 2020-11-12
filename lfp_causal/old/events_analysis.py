import numpy as np
import pandas as pd
from lfp_causal.old.smr2mne import smr_to_raw

# fname = 'D:\\Databases\\db_lfp\\lfp_causal\\freddie\\easy\\spike2\\fneu1098.xlsx'
def events_finder(events, fname):
    items = ['cue_onset', 'cue_offset', 'trigger_onset', 'trigger_offset',
             'movement_onset',  'reward_onset', 'reward_delivery', 'movement_offset']
    cue = np.intersect1d(events['cue L']['times'], events['cue R']['times'])
    cues = np.array([])
    for c1 in cue:
        for c2 in cue:
            if 0.499 < c2 - c1 < 0.501:
                cues = np.hstack((cues, np.array([c1, c2])))
    cue = cues
    cue = np.reshape(cue, (int(len(cue)/2), 2))
    assert np.all((cue[:, 1]-cue[:, 0]) > 0.499) \
           and np.all((cue[:, 1]-cue[:, 0]) < 0.501), \
           'not all cues are lasting around 0.5s'

    trigger = np.intersect1d(events['trig L']['times'],
                             events['trig R']['times'])
    try:
        trigger = np.reshape(trigger, (int(len(trigger)/2), 2))
        cue - trigger
    except:
        trigoff = np.intersect1d(events['bar']['times'], trigger)
        trigon = np.setdiff1d(trigger, trigoff)
        if len(trigoff) < len(trigon):
            trigoff = np.hstack((trigoff, trigon[-1]+0.7))
        trigson = np.array([])
        for t in trigon:
            for c in cue[:, 1]:
                if 0.99 < t - c < 1.01:
                    trigson = np.hstack((trigson, np.array(t)))
        trigon = trigson
        if len(trigon) < len(trigoff):
            _del = np.array([])
            for t1 in trigoff:
                for t2 in trigoff:
                    if 0 < t2 - t1 < 2.:
                        _del = np.hstack((_del, np.array(t2)))
            trigoff = np.setdiff1d(trigoff, _del)
        if len(trigon) < len(trigoff):
            for t, i in zip(trigon, range(len(trigon))):
                if trigoff[i] < t: trigoff = np.delete(trigoff, i)

        trigger = np.zeros((len(trigon), 2))
        trigger[:, 0] = trigon
        trigger[:, 1] = trigoff
    if trigger[-1, 0] < cue[-1, 1]:
        cue = np.delete(cue, -1, axis=0)
    while trigger.shape[0] != cue.shape[0]:
        if trigger.shape[0] < cue.shape[0]:
            # n_miss = cue.shape[0] - trigger.shape[0]
            # trigger = np.vstack((trigger, np.zeros((n_miss, 2))))
            for t, i in zip(trigger[:, 0], range(len(cue))):
                if t - cue[i, 1] >= 1.5:
                    cue = np.delete(cue, i, axis=0)

    cue_onset = cue[:, 0]
    cue_offset = cue[:, 1]

    # assert np.all(trigger[:, 1] - trigger[:, 0] <= 1)
    trigger_onset = trigger[:, 0]
    trigger_offset = trigger[:, 1]
    assert np.all((trigger_onset - cue_offset) > (1 - 0.01)) \
           and np.all((trigger_onset - cue_offset) < (1 + 0.01)),\
           'the distance between cues and trigger is incorrect'

    reward = np.intersect1d(events['bar']['times'], trigger)
    reward_onset = np.intersect1d(reward, trigger_offset)

    delivery = events['reward']['times']
    delivery = np.unique(np.around(delivery, decimals=0))
    _rd = np.array([])
    for r in reward_onset:
        for d in delivery:
            if r - 0.5 < d < r + 0.5:
                _rd = np.hstack((_rd, r)) #if '_rd' in locals() else np.array([r])
    _, i_rd, _ = np.intersect1d(reward_onset, _rd, return_indices=True)
    reward_delivery = np.zeros(len(reward_onset))
    np.put(reward_delivery, i_rd, 1)
    assert len(reward_onset) == len(reward_delivery), \
        'the numbers of rewards and their corresponding times are not the same'

    bar = events['bar']['times']
    bar = np.unique(bar)
    bar = np.delete(bar, np.where(bar < cue_onset[0])[0])
    bar = np.delete(bar, np.where(np.isin(bar, np.intersect1d(bar, cue.flatten())))[0])
    # bar = np.delete(bar, np.where(np.isin(bar, np.intersect1d(bar, trigger.flatten())))[0])
    bar = np.delete(bar,
                    np.where(
                        np.isin(
                            np.around(bar, decimals=1),
                            np.intersect1d(np.around(bar, decimals=1),
                                           np.around(trigger.flatten(),
                                                     decimals=1))))[0])

    movement_onset = np.array([])
    for x, y in zip(trigger[:, 0], trigger[:, 1]):
        for b in bar:
            if x < b < y:
                movement_onset = np.hstack((movement_onset, b))

    movement_offset = np.delete(bar, np.where(np.isin(bar, movement_onset))[0])
    _, i_mo = np.unique(np.around(movement_offset), return_index=True)
    movement_offset = np.take(movement_offset, i_mo)
    if movement_offset[-1] < trigger_offset[-1]:
        movement_offset = np.hstack((movement_offset, trigger_offset[-1] + 2.))
    while len(trigger_offset) != len(movement_offset):
        for t, i in zip(trigger_offset, range(len(trigger_offset))):
            if movement_offset[i] - t < 0:
                movement_offset = np.delete(movement_offset, i)
        if len(trigger_offset) < len(movement_offset):
            movement_offset = np.delete(movement_offset, -1)

    all_events = [cue_onset, cue_offset, trigger_onset, trigger_offset,
                  movement_onset,
                  reward_onset, reward_delivery, movement_offset]

    if len(reward_delivery) != len(trigger_offset):
        for i in range(len(all_events)):
            while len(all_events[i]) != len(reward_delivery):
                all_events[i] = np.delete(all_events[i], -1)

    d_ev = {i: e for i, e in zip(items, all_events)}
    df = pd.DataFrame.from_dict(d_ev)
    if fname.endswith('.smr'):
        fname = fname.replace('.smr', '.xlsx')
    df.to_excel(fname, sheet_name='events', index=False)

if __name__ == '__main__':
    import os
    files = []
    for file in os.listdir('D:\\Databases\\db_lfp\\lfp_causal\\freddie\\easy\\spike2'):
        if file.endswith('1165.smr'):
            fname = os.path.join('D:\\Databases\\db_lfp\\lfp_causal\\freddie\\easy\\spike2', file)
            smr = smr_to_raw(fname)
            events = smr[0][1]
            print('Processing events for {0}...'.format(file))
            events_finder(events, fname)
            print('...Done \n')
    # smr = smr_to_raw('D:\\Databases\\db_lfp\\lfp_causal\\freddie\\easy\\spike2\\fneu1098.smr')
    # events = smr[0][1]
    # events_finder(events)
