from lfp_causal.IO import read_smr
import numpy as np
import mne


def get_signal(data):
    signals_data = []
    for seg in data.segments:
        seg_data = {}
        for asig in seg.analogsignals:
            seg_data[asig.name] = {}
            keys = ['signal', 'times', 'sfreq']
            values = [np.array(asig.rescale('V').magnitude.ravel()),
                      asig.times, asig.sampling_rate]
            for k, v in zip(keys, values):
                seg_data[asig.name][k] = np.array(v)
        signals_data.append(seg_data)
    return signals_data


def smr_to_raw(fname):
    data = read_smr(fname)
    segments = get_signal(data)
    # events = get_events(data)

    raws = []
    for seg in segments:
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

        if raw.info['sfreq'] == 16667.0:
            print('Resampling at 12500.0 Hz...')
            raw.resample(12500, n_jobs=-1)

        raws.append(raw)

    for r in raws:
        fname = fname.replace('fneu', '')
        fname = fname.replace('tneu', '')
        fname = fname.replace('.smr', '_raw.fif')
        fname = fname.replace('/smr/', '/raw/')
        r.save(fname, overwrite=True)

    return raws


if __name__ == '__main__':
    import os

    monkey = 'teddy'
    condition = 'cued'

    smr_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
              '{0}/{1}/smr'.format(monkey, condition)

    files = []
    for file in os.listdir(smr_dir):
        if file.endswith('.smr'):
            fname = os.path.join(smr_dir, file)
            smr_to_raw(fname)
