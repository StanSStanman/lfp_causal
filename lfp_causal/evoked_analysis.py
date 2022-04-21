import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import zscore
from lfp_causal.old.controls import session_name, check_area
from lfp_causal.directories import epochs_dir


def collect_evoked(subject, condition, session, event='trigger', time_window=[-0.5, 0.5], picks=None):
    # cue_interval = [-2.0, -1.0]
    # trigger_interval = [-0.5, 0.5]

    # Correct session name and read the associate epochs file
    trial_num = session_name(session)
    epochs_fname = epochs_dir.format(subject, condition, trial_num) + '{0}_{1}-epo.fif'.format(trial_num, event)
    epochs = mne.read_epochs(epochs_fname, preload=True)

    if isinstance(picks, list):
        epochs.pick_channels(picks)

    # cue_epochs = epochs.copy().crop(cue_interval[0], cue_interval[1])
    # trigger_epochs = epochs.copy().crop(trigger_interval[0], trigger_interval[1])
    event_epochs = epochs.copy().crop(time_window[0], time_window[1])


    # cue_evoked = cue_epochs.copy().average()
    # cue_sem = cue_epochs.copy().standard_error()
    # trigger_evoked = trigger_epochs.copy().average()
    # trigger_sem = trigger_epochs.copy().standard_error()
    event_evoked = event_epochs.copy().average()
    event_sem = event_epochs.copy().standard_error()

    # fig = mne.viz.plot_compare_evokeds(cue_evoked, picks=[0], vlines=[-1.5])
    # return cue_evoked, cue_sem, trigger_evoked, trigger_sem
    return event_evoked, event_sem

def collect_avg_evoked(subject, condition, session, area, event='trigger', time_window=[-0.5, 0.5], picks=None):
    all_epochs = []
    for ses in session:
        trial_n = check_area(subject, condition, ses, area)
        if isinstance(trial_n, str):

            trial_num = session_name(ses)
            epochs_fname = epochs_dir.format(subject, condition, trial_num) + '{0}_{1}-epo.fif'.format(trial_num, event)
            epochs = mne.read_epochs(epochs_fname, preload=True)

            if isinstance(picks, list):
                epochs.pick_channels(picks)

            event_epochs = epochs.copy().crop(time_window[0], time_window[1])

            all_epochs.append(event_epochs)

    all_epochs = mne.concatenate_epochs(all_epochs)

    event_evoked = all_epochs.copy().average()
    event_sem = all_epochs.copy().standard_error()

    return event_evoked, event_sem


def plot_evoked(subject, condition, session, events_struct, picks=None, aligned=True, show=True):
    cm = plt.get_cmap('Set1')
    col = 0.11
    # fig, ax = plt.subplots()
    if show == True: fig = plt.figure()
    for k in events_struct.keys():
        time_window = events_struct[k][0]
        event = events_struct[k][1]
        event_evoked, event_sem = collect_evoked(subject, condition, session, event, time_window, picks)
        times = event_evoked.times
        if aligned == True: times -= np.average(times)
        average = event_evoked.data.squeeze() * 1000
        error = event_sem.data.squeeze() * 1000
        plt.plot(times, average, color=cm(col), label=k)
        plt.fill_between(times, average-error, average+error, color=cm(col), alpha=0.2)
        plt.axvline(np.average(times), color='k', linestyle=':')
        plt.axhline(0, color='k')
        col += 0.11
    if show == True:
        plt.title('Average evoked {0}'.format(session), fontsize=15)
        plt.xlabel('Time')
        plt.ylabel('mV')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # return ax

def plot_area_evoked(subject, condition, session, area, events_struct, picks=['lfp'], mode='all'):
    good_trials = []
    for ses in session:
        trial_n = check_area(subject, condition, ses, area)
        if isinstance(trial_n, str):
            good_trials.append(trial_n)


    if mode == 'all':
        fig = plt.figure()
        fig.suptitle(area, fontsize=18)
        for t in range(len(good_trials)):
            ax = fig.add_subplot(np.ceil(len(good_trials)/2.), 2, t+1)
            ax = plot_evoked(subject, condition, good_trials[t], events_struct, picks=picks, aligned=True, show=False)
            plt.title('Average evoked {0}'.format(good_trials[t]), fontsize=15)
            plt.xlabel('Time')
            plt.ylabel('mV')
            plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    elif mode == 'single':
        for t in range(len(good_trials)):
            plot_evoked(subject, condition, good_trials[t], events_struct, picks=picks, aligned=True, show=True)



def plot_avg_evoked(subject, condition, session, area, events_struct, picks=None, aligned=True):
    cm = plt.get_cmap('Set1')
    col = 0.11
    for k in events_struct.keys():
        time_window = events_struct[k][0]
        event = events_struct[k][1]
        event_evoked, event_sem = collect_avg_evoked(subject, condition, session, area, event, time_window, picks)
        times = event_evoked.times
        if aligned == True: times -= np.average(times)
        average = event_evoked.data.squeeze() * 1000
        error = event_sem.data.squeeze() * 1000
        plt.plot(times, average, color=cm(col), label=k)
        plt.fill_between(times, average-error, average+error, color=cm(col), alpha=0.2)
        plt.axvline(np.average(times), color='k', linestyle=':')
        plt.axhline(0, color='k')
        col += 0.11
        plt.title('Average evoked, all session, area {0}'.format(area), fontsize=15)
        plt.xlabel('Time')
        plt.ylabel('mV')
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_zscore_evoked(subject, condition, session, area, events_struct, picks=None, mode='max'):
    good_trials = []
    for ses in session:
        trial_n = check_area(subject, condition, ses, area)
        if isinstance(trial_n, str):
            good_trials.append(trial_n)

    if mode == 'single':
        for t in range(len(good_trials)):
            for ev in events_struct.keys():
                evk_avg, evk_sem = collect_evoked(subject, condition, good_trials[t], events_struct[ev][1],
                                                  events_struct[ev][0], picks)
                times = evk_avg.times
                times -= np.average(times)
                evk_zscore = zscore(evk_avg.data.squeeze())
                plt.plot(times, evk_zscore, label=ev)
                plt.axvline(np.average(times), color='k', linestyle=':')
                plt.axhline(0, color='k')

            plt.title('z-scores evoked {0}'.format(good_trials[t]), fontsize=15)
            plt.xlabel('Time')
            plt.ylabel('zscore(V)')
            plt.legend()
            plt.tight_layout()
            plt.show()

    if mode == 'max':
        ev_mex_zscore = {key: [] for key in events_struct.keys()}
        for t in range(len(good_trials)):
            for ev in events_struct.keys():
                evk_avg, evk_sem = collect_evoked(subject, condition, good_trials[t], events_struct[ev][1],
                                                  events_struct[ev][0], picks)
                max_zscore = max(zscore(evk_avg.data.squeeze()))
                ev_mex_zscore[ev].append(max_zscore)
        for ev in ev_mex_zscore.keys():
            plt.hist(ev_mex_zscore[ev], bins=len(ev_mex_zscore[ev])/2, label=ev, alpha=0.5)
            plt.axvline(2.58, color='g', linestyle='-')
            # plt.title('Max z-scores for {0}, {1}'.format(ev, area), fontsize=15)
            plt.title('Max z-scores for {0}'.format(area), fontsize=15)
            plt.xlabel('max zscore')
            plt.ylabel('N')
            plt.tight_layout()
        plt.legend(loc='best')
        plt.show()

    # if mode == 'area':


            # plot_evoked(subject, condition, good_trials[t], events_struct, picks=picks, aligned=True, show=True)



if __name__ == '__main__':
    # collect_evoked('freddie', 'easy', 'fneu0437', picks=['lfp'])
    events_struct = {'trigger':([-0.5, 0.5], 'trigger'), 'cue':([-2.0, -1.0], 'trigger')}
    # plot_evoked('freddie', 'easy', 'fneu0437', events, picks=['lfp'])
    # plot_area_evoked('freddie', 'easy', ['fneu0437', 'fneu0772', 'fneu0773', 'fneu0779'],
    #                  'associative striatum', events_struct, mode='all')
    # plot_avg_evoked('freddie', 'easy', ['fneu0437', 'fneu0772', 'fneu0773', 'fneu0779'],
    #                  'associative striatum', events_struct, ['lfp'])
    plot_zscore_evoked('freddie', 'easy', ['fneu0437', 'fneu0772', 'fneu0773', 'fneu0779'],
                     'associative striatum', events_struct, picks=['lfp'], mode='single')