import neo
import mne
import numpy as np
# from events_analysis import events_finder


def read_smr(fname):
    reader = neo.io.Spike2IO(filename=fname, try_signal_grouping=False)
    data = reader.read(lazy=False)[0]
    return data


def get_signal(data):
    signals_data = []
    for seg in data.segments:
        seg_data = {}
        for asig in seg.analogsignals:
            seg_data[asig.name] = {}
            keys = ['signal', 'times', 'sfreq']
            values = [np.array(asig.rescale('V').magnitude.ravel()), asig.times, asig.sampling_rate]
            for k, v in zip(keys, values):
                seg_data[asig.name][k] = np.array(v)
        signals_data.append(seg_data)
    return signals_data


def get_events(data):
    events_data = []
    for seg in data.segments:
        seg_data = {}
        for eve in seg.events:
            seg_data[eve.name] = {}
            keys = ['times', 'labels']
            values = [eve.times, eve.labels]
            for k, v in zip(keys, values):
                seg_data[eve.name][k] = np.array(v)
        events_data.append(seg_data)
    return events_data


def smr_to_raw(fname):
    data = read_smr(fname)
    segments = get_signal(data)
    events = get_events(data)

    raws = []
    for seg, eve in zip(segments, events):
        ch_names = list(seg.keys())
        for cn, i in zip(ch_names, range(len(ch_names))):
            if 'Channel bundle (' in cn:
                cn = cn.replace('Channel bundle (', '')
            if ') ' in cn:
                cn = cn.replace(') ', '')
            ch_names[i] = cn
        ch_types = 'seeg'
        # assert
        sfreq = np.round(float(seg[list(seg.keys())[0]]['sfreq']))

        signal =[]
        for k in seg.keys():
            signal.append(seg[k]['signal'])
        max_len = np.max([len(s) for s in signal])
        for s, i in zip(signal, range(len(signal))):
            signal[i] = np.pad(s, (0, max_len - len(s)), 'mean')
        signal = np.array(signal)

        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(signal, info)
        raws.append([raw, eve])
    return raws


if __name__ == '__main__':
    # smr_to_raw('D:\\Databases\\db_lfp\\lfp_causal\\freddie\\easy\\spike2\\fneu1028.smr')
    import os
    files = []
    for file in os.listdir('/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/'
                           'freddie/easy/smr'):
        if file.endswith('.smr'):
            fname = os.path.join('/media/jerry/TOSHIBA EXT/data/db_lfp/'
                                 'lfp_causal/freddie/easy/smr', file)
            smr = smr_to_raw(fname)
            raw = smr[0][0]
            file = file.replace('fneu', '')
            file = file.replace('.smr', '_raw.fif')
            raw.save(os.path.join('/media/jerry/TOSHIBA EXT/data/db_lfp/'
                                  'lfp_causal/freddie/easy/raw', file),
                     overwrite=True)
