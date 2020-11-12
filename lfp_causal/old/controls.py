import warnings
import logging
import pandas as pd
from lfp_causal.directories import *
from lfp_causal.old.read_infos import trial_info

def session_name(session):
    ''' Correct the session name in input

    :param session: str, name of the session
    :return: str, the number of the session
    '''
    if 'fneu' in session:
        trial_num = session.replace('fneu', '')
    elif 'fneu' not in session:
        trial_num = session
    else:
        raise Exception('File name not correct')
    return trial_num

def check_rejected_epochs(events_idx, dropped_idx, times, t_window):
    warnings.simplefilter('always', UserWarning)
    max_t = times[-1]
    # drop = np.delete(events_idx.T, dropped_idx.T)
    drop = [x for x in events_idx.tolist() if x not in dropped_idx.tolist()]
    if len(drop) == 0:
        return
    else:
        for d in drop:
            if d <= len(times):
                d_time = times[d]
            else: d_time = max_t + 1
            if (d_time + t_window[0]) < 0:
                logging.warning('Epochs rejected for time issue (t_min < 0)')
            elif (d_time + t_window[1] > max_t):
                warnings.warn('Epochs rejected for time issue (t_max > 0)')
            else: assert False, 'Ther is a problem in time series, please check.'

def check_area(subject, condition, session, areas):
    trial_num = session_name(session)
    infos = trial_info(subject, condition, trial_num)
    if infos['territory'] == areas:
        return trial_num
    else: return

def quality_check(subject, condition, min_quality=1):
    df = pd.read_excel(os.path.join(directory, subject, condition, 'LFP_quality_{0}.xlsx'.format(subject)), dtype={'file': str})
    good_files = df.loc[df['quality'] <= min_quality]
    good_sessions = good_files['file'].to_list()

    return good_sessions