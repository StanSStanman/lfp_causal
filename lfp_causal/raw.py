import numpy as np
import mne
from lfp_causal.read_infos import trial_info, read_matfile
from lfp_causal.directories import raw_dir
from lfp_causal.controls import session_name

def create_rawfiles(subject, condition, session):
    ''' Create and save the raw files

    :param subject: str, name of the subject
    :param condition: str, name of the condition
    :param session: str, name of the session
    :return: save a raw file as 'session_raw.fif' in the 'raw_dir' folder
    '''
    # Correct the session name
    trial_num = session_name(session)

    # Read the associated informatio in the mat file
    data_dict = read_matfile(subject, condition, session)

    # Create the channels matrix for the row object, and add a zero before because the starting time point in a raw object is zero
    time = data_dict['time'][0].astype(float)
    neu_data = np.vstack((data_dict['lfp'].astype(float), data_dict['mua'].astype(float)))#, data_dict['time']))
    zero = np.zeros((2, 1))
    neu_data = np.hstack((zero, neu_data))

    # Setting the insofrations for the raw object
    ch_types = ['seeg', 'seeg']
    ch_names = ['lfp', 'mua']
    # sfreq = len(time) / (time[-1] - time[0]) # I prefer to calculate the sfreq from data to have more accuracy, otherwise set to 16667.0
    if 12000.0 <= len(neu_data.T)/time[-1] < 13000.0:
        sfreq = 12500.0
    else:
    #     # sfreq = 16667.0
        sfreq = 16667.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    # Creating the raw object
    raw = mne.io.RawArray(neu_data, info)

    assert time[-1]-0.1 < raw.times[-1] < time[-1]+0.1, 'Error: too much time difference'
    # Save the raw object
    raw.save(raw_dir.format(subject, condition) + '{0}_raw.fif'.format(trial_num), overwrite=True)

def plot_rawdata(subject, condition, session, scale=2.):

    # Correct session name and read the associate raw file
    trial_num = session_name(session)
    raw_fname = raw_dir.format(subject, condition) + '{0}_raw.fif'.format(trial_num)
    raw = mne.io.read_raw_fif(raw_fname, preload=True)

    scalings = {'seeg': scale}
    raw.plot(n_channels=2, scalings=scalings, show=True, block=True)