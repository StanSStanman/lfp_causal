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


def find_events_txt(fname_in, fname_out, fname_txt):
    print(fname_in)
    data = read_smr(fname_in)
    orig_eve = get_events(data)
    txt_info = read_txt(fname_txt)

    cue_delay = np.array(txt_info['I-delai']).astype(float)
    cue_length = np.array(txt_info['I-durée']).astype(float)
    trig_delay = np.array(txt_info['PP']).astype(float)
    react_time = np.array(txt_info['TR']).astype(float)
    cont_time = np.array(txt_info['TM']).astype(float)
    next_cue = np.array(txt_info['I-I-Es']).astype(float)
    button = np.array(txt_info['N-conta']).astype(float)
    cue_cue_dist = (cue_delay + cue_length + trig_delay + \
                   react_time + cont_time + np.roll(next_cue, -1))[:-1]

    code_events = np.array(txt_info['code']).astype(int)

    for events in orig_eve:

        # Events to extract from data
        items = ['cue_on', 'cue_off', 'trig_on', 'mov_on',
                 'trig_off', 'reward', 'button']

        # Extracting cues time series (cue_on, cue_off):
        # If the two time points have a distance of about 0.5s
        # they're considered valid (error = +- 0.001s)
        # cue = np.intersect1d(events['cue L']['times'],
        #                      events['cue R']['times'])
        cue = np.unique(np.hstack((events['cue L']['times'],
                                   events['cue R']['times'])))
        cues = np.array([])
        for c1 in cue:
            for c2 in cue:
                if 0.498 < c2 - c1 < 0.502:
                    cues = np.hstack((cues, np.array([c1, c2])))
        cue = cues
        cue = np.reshape(cue, (int(len(cue)/2), 2))

        cue_onset = cue[:, 0]
        cue_offset = cue[:, 1]

        assert np.all((cue_offset - cue_onset) > 0.498) \
               and np.all((cue_offset - cue_onset) < 0.502), \
               'Not all cues are lasting around 0.5s'

        # Find if there is some lacking trial in txt or smr file
        missing_trials = []
        c_on = np.diff(cue_onset).round(3) * 1e3
        while len(cue_cue_dist) != len(c_on):
            if len(cue_cue_dist) > len(c_on):
                # c_on = np.diff(cue_onset).round(3) * 1e3
                for i, ccd in enumerate(cue_cue_dist):
                    if not -1. <= ccd - c_on[i] <= 1.:
                        cue_cue_dist = np.delete(cue_cue_dist, i)
                        missing_trials.append(i)
                        break
                    if i == len(c_on) - 1:
                        if np.allclose(cue_cue_dist[:len(c_on)], c_on, atol=1.):
                            missing_trials += list(range(len(c_on), len(cue_cue_dist)))
                            cue_cue_dist = np.delete(cue_cue_dist, list(range(len(c_on), len(cue_cue_dist))))
                        break
            elif len(c_on) > len(cue_cue_dist):
                for i, ccd in enumerate(cue_cue_dist):
                    if not -1. <= ccd - c_on[i] <= 1.:
                        cue_onset = np.delete(cue_onset, i + 1)
                        cue_offset = np.delete(cue_offset, i + 1)
                        c_on = np.diff(cue_onset).round(3) * 1e3
                        break
                    if i == len(cue_cue_dist) - 1:
                        if np.allclose(cue_cue_dist, c_on[:len(cue_cue_dist)], atol=1.):
                            cue_onset = cue_onset[:len(cue_cue_dist) + 1]
                            cue_offset = cue_offset[:len(cue_cue_dist) + 1]
                            c_on = np.diff(cue_onset).round(3) * 1e3
                        break

        # Cleaning all the other txt's vectors from missing trials
        if len(missing_trials) > 0:
            cue_delay = np.delete(cue_delay, np.array(missing_trials))
            cue_length = np.delete(cue_length, np.array(missing_trials))
            trig_delay = np.delete(trig_delay, np.array(missing_trials))
            react_time = np.delete(react_time, np.array(missing_trials))
            cont_time = np.delete(cont_time, np.array(missing_trials))
            next_cue = np.delete(next_cue, np.array(missing_trials))
            code_events = np.delete(code_events, np.array(missing_trials))
            button = np.delete(button, np.array(missing_trials))

        c_on = np.diff(cue_onset).round(3) * 1e3
        if not np.all(-1. <= (cue_cue_dist - c_on)) \
               and np.all((cue_cue_dist - c_on) <= 1.):
            warnings.warn('No correspondence between cue onset distance'
                          ' in text and smr file')
        # assert np.all(-1. <= (cue_cue_dist[:-1] - c_on)) \
        #        and np.all((cue_cue_dist[:-1] - c_on) <= 1.), \
        #        'No correspondence between cue onset distance ' \
        #        'in text and smr file'

        # Defining trigger onset vector
        # trigger = np.intersect1d(events['trig L']['times'],
        #                          events['trig R']['times'])
        trigger = np.unique(np.hstack((events['trig L']['times'],
                                       events['trig R']['times'])))

        trigger_onset = np.zeros(len(cue_onset))
        t_on = cue_offset + (trig_delay / 1e3)
        for i in range(len(trigger_onset)):
            t = np.where(np.logical_and(t_on[i] - 2e-3 <= trigger,
                                        trigger <= t_on[i] + 2e-3))
            if trigger[t].size != 0:
                trigger_onset[i] = trigger[t].mean()
            else:
                warnings.warn('Trigger onset value out of matrix')
                trigger_onset[i] = t_on[i]

        # Defining trigger offset vector
        trigger_offset = np.zeros(len(trigger_onset))
        t_off = trigger_onset + (react_time / 1e3) + (cont_time / 1e3)
        for i in range(len(trigger_offset)):
            t = np.where(np.logical_and(t_off[i] - 3e-3 <= trigger,
                                        trigger <= t_off[i] + 3e-3))
            if trigger[t].size != 0:
                trigger_offset[i] = trigger[t].mean()
            else:
                warnings.warn('Trigger offset value out of matrix')
                trigger_offset[i] = t_off[i]

        test_tr_off = cue_onset + ((cue_length + trig_delay +
                                    react_time + cont_time) / 1000)
        if not np.all(np.isclose(trigger_offset, test_tr_off, atol=0.001)):
            assert np.all(np.isclose((trigger_offset - trigger_onset),
                                     ((react_time + cont_time) / 1000.),
                                     atol=0.003)), \
                AssertionError('Triggers are not correctly aligned.')

        # Defining reward vector from txt's codes
        reward = np.zeros(len(cue_onset))
        reward[code_events == 0] = 1
        reward[code_events != 0] = 0

        # Defining movement onset vector by txt's code
        movements = events['bar']['times']
        movement_onset = np.zeros(len(cue_onset))
        movement_onset[code_events == 5400] = np.nan
        assert np.all((trigger_onset + (react_time / 1e3)) -
                      (trigger_offset - (cont_time / 1e3)) < 2e-3), \
            'Time distances between trigger onset/offset and ' \
            'movement onset are incorrect'
        m_on = trigger_onset + (react_time / 1e3)
        for i in range(len(movement_onset)):
            if not np.isnan(movement_onset[i]):
                m = np.where(np.logical_and(m_on[i] - 1e-3 <= movements,
                                            movements <= m_on[i] + 1e-3))
                if movements[m].size != 0:
                    # print(movements[m])
                    movement_onset[i] = movements[m].mean()
                elif movements[m].size == 0 and i == len(movement_onset) - 1:
                    warnings.warn('Movement onset value out of matrix')
                    movement_onset[i] = m_on[i]
                elif movements[m].size == 0 and i != len(movement_onset) - 1:
                    warnings.warn('Inferring movement onset value from txt')
                    movement_onset[i] = m_on[i]

        # Defining sequence of button press
        button[button == 0] = np.nan
        button[button != 0] += 100

        # Constructing the complete information xarray and writing on .csv file
        data = np.vstack((cue_onset, cue_offset, trigger_onset, movement_onset,
                          trigger_offset, reward, button)).T

        # Removing trials with no movement or button pression
        # (codes: 5400, 5500, 5510)
        no_movement = np.where(np.isnan(movement_onset))[0]
        no_button = np.where(np.isnan(button))[0]
        bad_trials = np.unique(np.hstack((no_movement, no_button)))
        data = np.delete(data, bad_trials, axis=0)
        print('Trials', list(bad_trials), 'removed because of NaNs')

        time_events = xr.DataArray(data=data,
                                   coords=[range(data.shape[0]), items],
                                   dims=['trials', 'events'])
        time_events.to_pandas().to_csv(fname_out)
        print(fname_out)

    return


def create_event_matrix(csv_fname, raw_fname, eve_dir):
    # Those two lists are here just with th purpose to remember the used codes
    events = ['cue_on', 'cue_off', 'trig_on', 'mov_on',
              'trig_off', 'reward', 'button']
    codes = [11, 10, 21, 31, 20, [0, 1], [102, 103, 104]]

    csv = pd.read_csv(csv_fname)
    # Read and clean csv infos and sfreq
    # csv, button_codes = check_events(csv_fname, txt_fname)
    sfreq = read_sfreq(raw_fname)

    # Define event matrix
    event_matrix = np.array(([], [], [])).reshape(0, 3)

    # Including base events
    eve = ['cue_on', 'cue_off', 'trig_on', 'mov_on', 'trig_off']
    cod = [11, 10, 21, 31, 20]
    for e, c in zip(eve, cod):
        _eve = (np.array(csv[e]) * sfreq).round()
        _cod = np.full(len(_eve), c)
        if e == 'trig_off':
            _zer = np.array(csv['reward'])
        else:
            _zer = np.zeros(len(_eve))
        _em = np.vstack((_eve, _zer, _cod)).T
        _em = np.delete(_em, np.where(np.isnan(_em[:, 0]))[0], axis=0)
        event_matrix = np.vstack((event_matrix, _em)).astype(int)

    # # Including button press events
    # for o, n in zip([2, 3, 4], [102, 103, 104]):
    #     button_codes[button_codes == o] = n
    # button_m = np.vstack(((csv['trig_off'] * sfreq).round(),
    #                       np.zeros(len(button_codes)), button_codes)).T
    # button_m = np.delete(button_m, np.where(np.isnan(button_m[:, 0]))[0],
    #                      axis=0)
    # event_matrix = np.vstack((event_matrix, button_m.astype(int)))
    #
    # # Including reward events
    # rew_m = np.vstack(((csv['rew_time'] * sfreq).round(),
    #                    np.zeros(len(csv['reward'])), csv['reward'])).T
    # rew_m = np.delete(rew_m, np.where(np.isnan(rew_m[:, 0]))[0], axis=0)
    # event_matrix = np.vstack((event_matrix, rew_m.astype(int)))

    # Sorting event matrix
    event_matrix = event_matrix[np.argsort(event_matrix[:, 0])]

    # Saving event file
    fname_eve = raw_fname.split('/')[-1].replace('raw', 'eve')
    mne.write_events(os.path.join(eve_dir, fname_eve), event_matrix)

    return


if __name__ == '__main__':
    import os

    monkey = 'teddy'
    condition = 'cued'
    label = 'tneu'

    # rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
    #            '{0}/{1}/files_info.xlsx'.format(monkey, condition)
    # xls = pd.read_excel(rec_info, dtype={'file': str})
    # new_files = xls['file']
    # start = np.where(new_files == '0219')[0][0]
    # stop = len(new_files)
    # new_files = new_files[start:stop].to_list()

    rej = ['tneu0458.smr', 'tneu0460.smr', 'tneu0539.smr',  'tneu0290.smr']

    files = []
    for file in os.listdir('/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/'
                           '{0}/{1}/smr'.format(monkey, condition)):
        if file in rej:
            continue

        # session = file.replace('smr', '').replace(label, '')
        # if session not in new_files:
        #     continue

        # file = 'tneu0743.smr'
        if file.endswith('.smr'):
            fname_in = os.path.join('/media/jerry/TOSHIBA EXT/data/db_lfp/'
                                    'lfp_causal/'
                                    '{0}/{1}/smr'.format(monkey, condition),
                                    file)

            file_out = file.replace(label, '').replace('.smr', '.csv')
            fname_out = os.path.join('/media/jerry/TOSHIBA EXT/data/'
                                     'db_behaviour/lfp_causal/'
                                     '{0}/{1}/'.format(monkey, condition),
                                     't_events', file_out)

            fname_txt = os.path.join('/media/jerry/TOSHIBA EXT/data/'
                                     'db_behaviour/lfp_causal/'
                                     '{0}/{1}/'.format(monkey, condition),
                                     'info', file.replace('.smr', '.txt'))

            find_events_txt(fname_in, fname_out, fname_txt)
            print(file, 'done')

            file_raw = file_out.replace(label, '')
            file_raw = file_raw.replace('.csv', '_raw.fif')
            fname_raw = os.path.join('/media/jerry/TOSHIBA EXT/data/'
                                     'db_lfp/lfp_causal/'
                                     '{0}/{1}/raw'.format(monkey, condition),
                                     file_raw)
            eve_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal' \
                      '/{0}/{1}/eve'.format(monkey, condition)
            create_event_matrix(fname_out, fname_raw, eve_dir)
