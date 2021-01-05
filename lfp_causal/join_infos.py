import numpy as np
import pandas as pd

def join_infos(monkey, condition):
    fname_rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
                   '{0}/{1}/recording_info.xlsx'.format(monkey, condition)
    fname_good_ch = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
                  '{0}/{1}/good_lfp.xlsx'.format(monkey, condition)
    fname_bad_epo = '/media/jerry/TOSHIBA EXT/data/db_behaviour/lfp_causal/' \
                  '{0}/{1}/lfp_bad_trials.csv'.format(monkey, condition)

    xls_rec_info = pd.read_excel(fname_rec_info,
                                 dtype={'monkey': str,
                                        'file': str,
                                        'neuron': float,
                                        'neuron_type': str,
                                        'record': float,
                                        'sector': str,
                                        'test': str,
                                        'target_location': str,
                                        'good_channel': float,
                                        'quality': float,
                                        'AP': str,
                                        'ML': str,
                                        'depth': float})
    xls_good_ch = pd.read_excel(fname_good_ch,
                                dtype={'file': str,
                                       'good_channel': float})
    csv_bad_epo = pd.read_csv(fname_bad_epo,
                              dtype=str)

    new_info = {'monkey': [],
                'file': [],
                'neuron': [],
                'neuron_type': [],
                'sector': [],
                'test': [],
                'target_location': [],
                'good_channel': [],
                'quality': [],
                'bad_LFP1': [],
                'bad_LFP2': [],
                'AP': [],
                'ML': [],
                'depth': []}

    sessions = np.unique(np.array(xls_rec_info['file'], dtype=str))

    for ses in sessions:
        new_info['monkey'].append(xls_rec_info['monkey']
                                  [xls_rec_info['file'] == ses].values[0])
        new_info['file'].append(ses)
        new_info['neuron'].append(xls_rec_info['neuron']
                                  [xls_rec_info['file'] == ses].values[0])
        new_info['neuron_type'].append(xls_rec_info['neuron_type']
                                       [xls_rec_info['file'] == ses].values[0])
        new_info['sector'].append(xls_rec_info['sector']
                                  [xls_rec_info['file'] == ses].values[0])
        new_info['test'].append(xls_rec_info['test']
                                [xls_rec_info['file'] == ses].values[0])
        new_info['target_location'].append(xls_rec_info['target_location']
                                           [xls_rec_info['file'] == ses]
                                           .values[0])

        if ses in np.array(xls_good_ch['file'], dtype=str):
            new_info['good_channel'].append(xls_good_ch['good_channel']
                                            [xls_good_ch['file'] == ses]
                                            .values[0])
        else:
            new_info['good_channel'].append(xls_rec_info['good_channel']
                                            [xls_rec_info['file'] == ses]
                                            .values[0])

        new_info['quality'].append(xls_rec_info['quality']
                                   [xls_rec_info['file'] == ses].values[0])

        if ses in np.array(csv_bad_epo['session'], dtype=str):
            bad_lfp1 = (csv_bad_epo['LFP1']
            [csv_bad_epo['session'] == ses].values[0])

            bad_lfp2 = (csv_bad_epo['LFP2']
            [csv_bad_epo['session'] == ses].values[0])

            bads = []
            for b in [bad_lfp1, bad_lfp2]:
                if b == 'None':
                    bads.append([])
                else:
                    bads.append(list(map(int, b.split(','))))
        else:
            bads = [[], []]

        new_info['bad_LFP1'].append(bads[0])
        new_info['bad_LFP2'].append(bads[1])
        new_info['AP'].append(xls_rec_info['AP']
                              [xls_rec_info['file'] == ses].values[0])
        new_info['ML'].append(xls_rec_info['ML']
                              [xls_rec_info['file'] == ses].values[0])
        new_info['depth'].append(xls_rec_info['depth']
                                 [xls_rec_info['file'] == ses].values[0])

    df = pd.DataFrame.from_dict(new_info)
    df.to_excel('/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
                   '{0}/{1}/files_info.xlsx'.format(monkey, condition),
                sheet_name=monkey, index=False)
    return


if __name__ == '__main__':
    monkey = 'freddie'
    condition = 'hard'

    join_infos(monkey, condition)
