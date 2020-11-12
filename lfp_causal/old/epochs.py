import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from lfp_causal.directories import raw_dir, epochs_dir
from lfp_causal.old.read_infos import read_matfile
from lfp_causal.old.controls import session_name, check_rejected_epochs

def create_epochs(subject, condition, session):
    ''' Create and save the epochs files aligned on trigger, action and outcome

    :param subject: str, name of the subject
    :param condition: str, name of the condition
    :param session: str, name of the session
    :return: save epochs files in different folders named by session
    '''

    # Here one can define the time windows to take around each event
    t_times = [-2.5, 1.5]
    a_times = [-1.5, 1.5]
    o_times = [-1.5, 1.5]

    # Here one can define the baseline to calculate each epochs
    t_bline = (-2.0, -1.5)
    a_bline = (None, None)
    o_bline = (None, None)

    # Correct session name and read the associate raw file
    trial_num = session_name(session)
    raw_fname = raw_dir.format(subject, condition) + '{0}_raw.fif'.format(trial_num)
    raw = mne.io.read_raw_fif(raw_fname)

    if raw.info['sfreq'] == 16667.0:
        raw.load_data()
        raw.resample(12500, n_jobs=-1)
        raw.close()

    raw.notch_filter(np.arange(50, 251, 50))

    # Read the associated information in the mat file
    data_dict = read_matfile(subject, condition, session)

    # Save events files in xls format
    df = pd.DataFrame.from_dict({'trigger_time': data_dict['trigger_time'].flatten().tolist(),
                                 'action_time': data_dict['action_time'].flatten().tolist(),
                                 'contact_time': data_dict['contact_time'].flatten().tolist()})
    df.to_excel(os.path.join('D:\\Databases\\db_lfp\\lfp_causal\\freddie\\easy\\events', trial_num+'.xlsx'),
                sheet_name='events', index=False)

    # Searching the index correspondent to each event
    trig_times = data_dict['trigger_time'].copy()
    trig_times = np.round(trig_times * raw.info['sfreq']).astype(int)
    act_times = data_dict['action_time'].copy()
    act_times = np.round(act_times * raw.info['sfreq']).astype(int)
    outc_times = data_dict['contact_time'].copy()
    outc_times = np.round(outc_times * raw.info['sfreq']).astype(int)

    # Building events matrices
    trigger_events = np.hstack((trig_times, np.zeros((len(trig_times), 1), dtype=int), np.ones((len(trig_times), 1), dtype=int)))
    action_events = np.hstack((act_times, np.zeros((len(act_times), 1), dtype=int), data_dict['action'].astype(int)))
    outcome_events = np.hstack((outc_times, np.zeros((len(outc_times), 1), dtype=int), data_dict['outcome'].astype(int)))

    # Creating the epochs object and rejecting bad epochs
    trig_epochs = mne.Epochs(raw, trigger_events, tmin=t_times[0], tmax=t_times[1], baseline=t_bline)
    trig_epochs.drop_bad()
    act_epochs = mne.Epochs(raw, action_events, tmin=a_times[0], tmax=a_times[1], baseline=a_bline)
    act_epochs.drop_bad()
    outc_epochs = mne.Epochs(raw, outcome_events, tmin=o_times[0], tmax=o_times[1], baseline=o_bline)
    outc_epochs.drop_bad()

    # Checking if the rejected epochs are out of time constraints
    for e, ee, tw in zip([trigger_events[:, 0], action_events[:, 0], outcome_events[:, 0]],
                         [trig_epochs.events[:, 0], act_epochs.events[:, 0], outc_epochs.events[:, 0]],
                         [t_times, a_times, o_times]):
        check_rejected_epochs(e, ee, raw.times, tw)

    # Creating folders for epochs
    epo_dir = epochs_dir.format(subject, condition, trial_num)# + '{0}\\'.format(trial_num)
    if not os.path.exists(epo_dir):
        os.makedirs(epo_dir)

    # Saving epochs objects
    trig_epochs.save(epo_dir + '{0}_trigger-epo.fif'.format(trial_num))
    act_epochs.save(epo_dir + '{0}_action-epo.fif'.format(trial_num))
    outc_epochs.save(epo_dir + '{0}_outcome-epo.fif'.format(trial_num))

    # Uncomment the following lines if you want to calculate the baseline too
    # baseline_epochs = mne.Epochs(raw, trigger_events, tmin=-2.0, tmax=-1.5, baseline=(None, None))
    # baseline_epochs.save(epo_dir + '{0}_baseline-epo.fif'.format(trial_num))

def plot_epochs(subject, condition, session, item, scale=2., n_epochs=5, picks=None, all=False):

    # Correct session name and read the associate epochs file
    trial_num = session_name(session)
    epochs_fname = epochs_dir.format(subject, condition, trial_num) + '{0}_{1}-epo.fif'.format(trial_num, item)
    epochs = mne.read_epochs(epochs_fname, preload=True)

    if isinstance(picks, list):
        epochs.pick_channels(picks)
    epochs.resample(1000)

    scalings = {'seeg': scale}
    if all == True:
        epochs.plot_image([0])
    elif all == False:
        epochs.plot(n_channels=2, scalings=scalings, n_epochs=n_epochs, show=True, block=True)
    plt.show()
