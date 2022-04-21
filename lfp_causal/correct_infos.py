import pandas as pd

def check_correct_infos(xls_info, xls_check):
    infos = pd.read_excel(xls_info, dtype={'file': str,
                                           'neuron': str,
                                           'AP': str,
                                           'ML': str,
                                           'depth': float})
    _infos = infos.copy()
    check = pd.read_excel(xls_check, dtype={'neuron': str,
                                           'AP': str,
                                           'ML': str,
                                           'new_AP': float,
                                           'new_ML': float,
                                           'depth': float})

    for i, neu in enumerate(infos['neuron']):
        # Name of the neuron in the check file
        if len(neu) == 2:
            ch_neu = 'N00' + neu
        elif len(neu) == 3:
            ch_neu = 'N0' + neu
        # Neuron relative info in check file
        ch_info = check[check['neuron'] == ch_neu]

        infos.at[i, 'AP'] = ch_info['new_AP'].values[0]
        infos.at[i, 'ML'] = ch_info['new_ML'].values[0]

        if ch_info['sector'].values[0] == 'striatum associatif':
            infos.at[i, 'sector'] = 'associative striatum'
        elif ch_info['sector'].values[0] == 'striatum limbique':
            infos.at[i, 'sector'] = 'limbic striatum'
        elif ch_info['sector'].values[0] == 'striatum moteur':
            infos.at[i, 'sector'] = 'motor striatum'

    infos.to_excel(xls_info, sheet_name='infos', index=False)
    _infos.to_excel(xls_info.replace('files_info', 'files_info_old'),
                    sheet_name='infos', index=False)

    return


if __name__ == '__main__':
    monkeys = ['freddie', 'teddy']
    conditions = ['easy', 'hard']

    for monkey in monkeys:
        fname_check = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
                      '{0}/{0}-dbase-coordinates.xlsx'.format(monkey)
        for condition in conditions:
            fname_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
                         '{0}/{1}/files_info.xlsx'.format(monkey, condition)

            check_correct_infos(fname_info, fname_check)

