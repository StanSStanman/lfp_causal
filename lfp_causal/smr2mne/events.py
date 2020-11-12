from lfp_causal.IO import read_smr, read_txt, read_sfreq
import numpy as np
import mne
import pandas as pd
import xarray as xr
import warnings


def get_events(data):
    events_data = []
    for seg in data.segments:
        seg_data = {}
        for eve in seg.events:
            if eve.name != '':
                seg_data[eve.name] = {}
                keys = ['times', 'labels']
                values = [eve.times, []]
                for k, v in zip(keys, values):
                    seg_data[eve.name][k] = np.array(v)
        seg_data['sfreq'] = float(seg.analogsignals[0].sampling_rate.round(3))
        events_data.append(seg_data)
    return events_data


def find_events(fname_in, fname_out):
    print(fname_in)
    data = read_smr(fname_in)
    orig_eve = get_events(data)

    for events in orig_eve:

        # Events to extract from data
        items = ['cue_on', 'cue_off', 'trig_on', 'trig_off',
                 'mov_on', 'mov_off', 'rew_time', 'reward']

        # Extracting cues time series (cue_on, cue_off):
        # If the two time points have a distance of about 0.5s
        # they're considered valid (error = +- 0.001s)
        cue = np.intersect1d(events['cue L']['times'], events['cue R']['times'])
        cues = np.array([])
        for c1 in cue:
            for c2 in cue:
                if 0.498 < c2 - c1 < 0.502:
                    cues = np.hstack((cues, np.array([c1, c2])))
        cue = cues
        cue = np.reshape(cue, (int(len(cue)/2), 2))
        assert np.all((cue[:, 1]-cue[:, 0]) > 0.498) \
               and np.all((cue[:, 1]-cue[:, 0]) < 0.502), \
               'not all cues are lasting around 0.5s'

        # Extracting trigger time series (trig_on, trig_off)
        # The trig_on event is distant 1s +- 0.01s from the cue_off
        # The trig_off event is following the trig_on (really relevant)
        trigger = np.intersect1d(events['trig L']['times'],
                                 events['trig R']['times'])
        trigger = np.setdiff1d(trigger, cue.flatten())
        # try:
        #     trigger = np.reshape(trigger, (int(len(trigger)/2), 2))
        # if True:
        # except:
        trigoff = np.intersect1d(events['bar']['times'], trigger)
        trigon = np.setdiff1d(trigger, trigoff)
        # if len(trigoff) < len(trigon):
        #     trigoff = np.hstack((trigoff, trigon[-1]+0.7))
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
                if trigoff[i] < t:
                    trigoff = np.delete(trigoff, i)

        trigger = np.zeros((len(trigon), 2))
        trigger[:, 0] = trigon
        trigger[:, 1] = trigoff
        if trigger[-1, 0] < cue[-1, 1]:
            cue = np.delete(cue, -1, axis=0)
        while trigger.shape[0] != cue.shape[0]:
            if trigger.shape[0] < cue.shape[0]:
                for t, i in zip(trigger[:, 0], range(len(cue))):
                    if t - cue[i, 1] >= 1.5:
                        cue = np.delete(cue, i, axis=0)
                        break
            if trigger.shape[0] > cue.shape[0]:
                for t, i in zip(trigger[:, 0], range(len(cue))):
                    if t - cue[i, 1] >= 1.005 or t - cue[i, 1] <= .995:
                        trigger = np.delete(trigger, i, axis=0)
                        break

        cue_onset = cue[:, 0]
        cue_offset = cue[:, 1]

        # assert np.all(trigger[:, 1] - trigger[:, 0] <= 1)
        trigger_onset = trigger[:, 0]
        trigger_offset = trigger[:, 1]
        assert np.all((trigger_onset - cue_offset) > (1 - 0.01)) \
               and np.all((trigger_onset - cue_offset) < (1 + 0.01)),\
               'the distance between cues and trigger is incorrect'

        # Extracting reward time series and value (rew_time, reward)
        # Beware of different pump release times
        # Information about not rewarded trials added later in the script
        reward = np.intersect1d(events['bar']['times'], trigger)
        reward_onset = np.intersect1d(reward, trigger_offset)

        delivery = events['reward']['times']
        # delivery = np.unique(np.around(delivery, decimals=0))
        _, count = np.unique(delivery.round(0), return_counts=True)
        while not np.all(count == 2):
            # _, count = np.unique(delivery.round(1), return_counts=True)
            if np.any(count < 2):
                ti = np.where(count < 2)[0][0]
                tp = np.sum(count[:ti])
                tv = delivery[tp]
                delivery = np.insert(delivery, tp, tv)
            _, count = np.unique(delivery.round(0), return_counts=True)
            if np.any(count > 2):
                ti = np.where(count > 2)[0][0]
                tp = np.sum(count[:ti])
                # tv = delivery[tp]
                delivery = np.delete(delivery, tp)
            _, count = np.unique(delivery.round(0), return_counts=True)

        delivery = delivery.reshape(int(len(delivery) / 2), 2).mean(1)
        pump_delays = [.13, .15, .18, .23]
        for pd in pump_delays:
            _rd = np.array([[], []]).T
            for r in reward_onset:
                _rr = 0
                for d in delivery:
                    if pd - 0.01 < d - r < pd + 0.01:
                        _rd = np.vstack((_rd, np.array([d, 1])))
                        _rr += 1
                if _rr == 0:
                    _rd = np.vstack((_rd, np.array([np.nan, 0])))

            if not np.all(np.isnan(_rd[:, 0])):
                break
        reward = _rd.copy()

        # Extracting movements time series (mov_on, mov_off)
        # mov_on is the first event after trig_on, as mov_off is the first
        # event after trig_off
        bar = events['bar']['times']
        bar = np.unique(bar)

        movement_onset = np.array([])
        for x, y in zip(trigger[:, 0], trigger[:, 1]):
            _m = 0
            _mo = np.array([])
            for b in bar:
                if x < b < y:
                    _mo = np.hstack((_mo, b))
                    # movement_onset = np.hstack((movement_onset, b))
                    _m += 1
            if _m == 0:
                movement_onset = np.hstack((movement_onset, np.nan))
            elif _m == 1:
                movement_onset = np.hstack((movement_onset, _mo))
            elif _m > 1:
                movement_onset = np.hstack((movement_onset, _mo[0]))
                warnings.warn('More than one movement between trigger on/off '
                              + '({0}), averaging times...'.format(_m))
                # raise ValueError('More than one movement during trig on/off')

        movement_offset = np.array([])
        _c1 = np.roll(cue[:, 0].copy(), -1)
        _c1[-1] = trigger[-1, 1] + 2
        for x, y in zip(trigger[:, 1], _c1):
            _mv = []
            _m = 0
            for b in bar:
                if x < b < y:
                    _mv.append(b)
                    # movement_offset = np.hstack((movement_offset, b))
                    _m += 1
            if _m == 0:
                movement_offset = np.hstack((movement_offset, np.nan))
            elif _m == 1:
                movement_offset = np.hstack((movement_offset, _mv[0]))
            elif _m > 1:
                ir = []
                for iv, v in enumerate(_mv):
                    if v - x < 0.5:
                        ir.append(iv)
                _mv = list(np.delete(np.array(_mv), ir, None))
                # for _ir in ir:
                #     _mv.pop(_ir)
                if not _mv:
                    movement_offset = np.hstack((movement_offset, np.nan))
                else:
                    movement_offset = np.hstack((movement_offset, _mv[0]))

                # raise ValueError('More than one movement during trigoff/cue')

        movement = np.hstack((np.expand_dims(movement_onset, 1),
                              np.expand_dims(movement_offset, 1)))

        # Integrating information about not rewarded trials
        # (no move or exceeded time)
        if len(reward) != len(trigger):
            no_contact = np.where(trigger_offset - movement_onset < .001)[0]
            if len(no_contact) > 0:
                for nc in no_contact:
                    reward = np.insert(reward, nc, [np.nan, 0.], axis=0)

            no_movement = np.where(
                np.isnan(trigger_offset - movement_onset))[0]
            if len(no_movement) > 0:
                for nm in no_movement:
                    reward = np.insert(reward, nm, [np.nan, 0.], axis=0)

        assert len(reward) == len(trigger), \
               'the numbers of rewards and triggers are not the same'

        # Constructing the complete information xarray and writing on .csv file
        time_events = xr.DataArray(
            data=np.hstack((cue, trigger, movement, reward)),
            coords=[range(cue.shape[0]), items], dims=['trials', 'events'])
        time_events.to_pandas().to_csv(fname_out)
        print(fname_out)

    return


def check_events(csv_fname, txt_fname):
    csv = pd.read_csv(csv_fname)
    txt = read_txt(txt_fname)

    rew_csv = np.array(csv['reward'])
    rew_txt = np.array(txt['N°-Rec']).astype(int)

    if len(rew_csv) == len(rew_txt):
        if np.all(rew_csv == rew_txt):
            button_codes = np.array(txt['N-conta']).astype(int)
            return csv, button_codes
        else:
            raise AssertionError('No correspondence between rewards')
    elif len(rew_csv) < len(rew_txt):
        for l in range(len(rew_txt)):
            rew_txt = np.roll(rew_txt, -1)
            if np.all(rew_txt[:len(rew_csv)] == rew_csv):
                button_codes = np.array(txt['N-conta'])[l + 1:].astype(int)
                return csv, button_codes
        raise AssertionError('Unable to align events')
    elif len(rew_csv) > len(rew_txt):
        raise AssertionError('There is a huge problem...')


def create_event_matrix(csv_fname, txt_fname, raw_fname, eve_dir):
    # Those two lists are here just with th purpose to remember the used codes
    events = ['cue_on', 'cue_off', 'trig_on', 'trig_off',
              'mov_on', 'mov_off', 'rew_time', 'reward']
    codes = [11, 10, 21, [102, 103, 104], 31, 30, [0, 1]]

    # Read and clean csv infos and sfreq
    csv, button_codes = check_events(csv_fname, txt_fname)
    sfreq = read_sfreq(raw_fname)

    # Define event matrix
    event_matrix = np.array(([], [], [])).reshape(0, 3)

    # Including base events
    eve = ['cue_on', 'cue_off', 'trig_on', 'mov_on', 'mov_off']
    cod = [11, 10, 21, 31, 30]
    for e, c in zip(eve, cod):
        _eve = (np.array(csv[e]) * sfreq).round()
        _cod = np.full(len(_eve), c)
        _zer = np.zeros(len(_eve))
        _em = np.vstack((_eve, _zer, _cod)).T
        _em = np.delete(_em, np.where(np.isnan(_em[:, 0]))[0], axis=0)
        event_matrix = np.vstack((event_matrix, _em)).astype(int)

    # Including button press events
    for o, n in zip([2, 3, 4], [102, 103, 104]):
        button_codes[button_codes == o] = n
    button_m = np.vstack(((csv['trig_off'] * sfreq).round(),
                          np.zeros(len(button_codes)), button_codes)).T
    button_m = np.delete(button_m, np.where(np.isnan(button_m[:, 0]))[0],
                         axis=0)
    event_matrix = np.vstack((event_matrix, button_m.astype(int)))

    # Including reward events
    rew_m = np.vstack(((csv['rew_time'] * sfreq).round(),
                       np.zeros(len(csv['reward'])), csv['reward'])).T
    rew_m = np.delete(rew_m, np.where(np.isnan(rew_m[:, 0]))[0], axis=0)
    event_matrix = np.vstack((event_matrix, rew_m.astype(int)))

    # Sorting event matrix
    event_matrix = event_matrix[np.argsort(event_matrix[:, 0])]

    # Saving event file
    fname_eve = raw_fname.split('/')[-1].replace('raw', 'eve')
    mne.write_events(os.path.join(eve_dir, fname_eve), event_matrix)

    return


if __name__ == '__main__':
    import os

    monkey = 'freddie'
    condition = 'easy'

    files = []
    for file in os.listdir('/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/'
                           '{0}/{1}/smr'.format(monkey, condition)):
        file = 'fneu0978.smr'
        if file.endswith('.smr'):
            fname_in = os.path.join('/media/jerry/TOSHIBA EXT/data/db_lfp/'
                                    'lfp_causal/'
                                    '{0}/{1}/smr'.format(monkey, condition),
                                    file)

            file_out = file.replace('.smr', '.csv')
            fname_out = os.path.join('/media/jerry/TOSHIBA EXT/data/'
                                     'db_behaviour/lfp_causal/'
                                     '{0}/{1}/'.format(monkey, condition),
                                     't_events', file_out)

            find_events(fname_in, fname_out)

            fname_txt = os.path.join('/media/jerry/TOSHIBA EXT/data/'
                                     'db_behaviour/lfp_causal/'
                                     '{0}/{1}/'.format(monkey, condition),
                                     'info', file_out.replace('.csv', '.txt'))
            file_raw = file_out.replace('fneu', '')
            file_raw = file_raw.replace('.csv', '_raw.fif')
            fname_raw = os.path.join('/media/jerry/TOSHIBA EXT/data/'
                                     'db_lfp/lfp_causal/'
                                     '{0}/{1}/raw'.format(monkey, condition),
                                     file_raw)
            eve_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal' \
                      '/{0}/{1}/eve'.format(monkey, condition)
            create_event_matrix(fname_out, fname_txt, fname_raw, eve_dir)
