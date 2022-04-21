import os
import pandas as pd
import numpy as np

def create_folders(directory, subject, condition):
    ''' Function to create all the folders needed for the analysis

    :param directory: str, main directory of the lfp database
    :param subject: list of str, name of the subjects
    :param condition: list of str, task condition

    :return: create the folders to store the data
    '''
    folders = ['infos', 'beh_data', 'neu_data', 'prep', 'epochs', 'raw']
    for sub in subject:
        for cond in condition:
            for f in folders:
                if not os.path.exists(directory + '{0}\\{1}\\{2}'.format(sub, cond, f)):
                    os.makedirs(directory + '{0}\\{1}\\{2}'.format(sub, cond, f))


def save_info(directory, xls, subject, condition, info_dir, rawmat_dir):
    ''' Function to read and save in different files all the information about each condition

    :param directory: str, main directory of the lfp database
    :param xls: str, name of the xls file cointaining the information
    :param subject: list of str, names of the subjects to iterate
    :param condition: list of str, conditions of the task to iterate
    :param info_dir: str, name of the directory in which the infos will be saved
    :param rawmat_dir: str, name of the directory containing the raw mat files
    :return: saves all the info in different json files (dict),
             save a file containing the name of the present and missing mat file (list of lists)
             save a file with the info labels (list)
    '''
    for sub in subject:
        for cond in condition:
            ld_xls = pd.read_excel(directory + xls.format(cond), sheet_name=sub)
            ld_xls = ld_xls.rename(columns={'Best block-target': 'block_target'})
            items = list(ld_xls.columns.values.astype(str))
            date = items.pop(1)
            items_form = [str, int, int, str, int, str, int, int, str, str, str]
            for i, fi in zip(items, items_form):
                ld_xls[i] = ld_xls[i].apply(fi)
            ld_xls[date] = ld_xls[date].dt.strftime('%d/%m/%Y')
            items.insert(1, date)
            # ld_xls['file'] = ld_xls['file'].apply(str)
            for f, n in zip(ld_xls['file'], range(ld_xls.shape[0])):
                if len(f) < 4:
                    for l in range(4 - len(f)):
                        f = '0' + f
                    ld_xls['file'][n] = f
            for s in ld_xls.iterrows():
                s[1].to_json(path_or_buf=info_dir.format(sub, cond) + 'info_{0}.json'.format(s[1]['file']),
                             orient='index', force_ascii=False)
                # s[1].to_excel(path_or_buf=info_dir.format(sub, cond) + 'info_{0}.xlsx'.format(s[1]['file']),
                #              sheet_name='events', index=False)
            files = ld_xls['file'].tolist()
            pres_f, abs_f = [], []
            for fn in files:
                if os.path.isfile(rawmat_dir.format(sub, cond) + 'fneu{0}.mat'.format(fn)):
                    pres_f.append('fneu'+fn)
                else: abs_f.append('fneu'+fn)
            print(len(abs_f), 'files not found for {0}, condition {1}:'.format(sub, cond), '\n', abs_f)
            np.save(info_dir.format(sub, cond) + 'files_info', [pres_f, abs_f])
            np.save(info_dir.format(sub, cond) + 'info_args', items)