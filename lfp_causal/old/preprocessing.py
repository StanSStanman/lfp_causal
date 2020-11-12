# matplotlib.use('Agg')
import numpy as np
from scipy.stats import zscore
import mne
import os
import matplotlib.pyplot as plt
from lfp_causal.old.controls import session_name, check_area
from lfp_causal.directories import plot_dir, raw_dir
from lfp_causal.old.read_infos import read_matfile


def preprocessing(subject, condition, session):
    trial_num = session_name(session)
    if not os.path.exists(plot_dir.format(subject,condition) + trial_num):
        os.mkdir(plot_dir.format(subject,condition) + trial_num)

    # Import raw files
    raw_fname = raw_dir.format(subject, condition) + '{0}_raw.fif'.format(trial_num)
    raw = mne.io.read_raw_fif(raw_fname, preload=True)

    # Band pass filter
    # raw.filter(1, 200)

    # Notch filter
    # raw.notch_filter(np.arange(50, 151, 50))

    # raw.plot_psd(fmax=200)

    # Taking session info on mat file
    t_times = [-2.5, 1.5]
    t_bline = (-2.0, -1.5)
    data_dict = read_matfile(subject, condition, session)
    trig_times = data_dict['trigger_time'].copy()
    trig_times = np.round(trig_times * raw.info['sfreq']).astype(int)
    acti_times = data_dict['action_time'].copy()
    acti_times = np.round(acti_times * raw.info['sfreq']).astype(int)
    outc_times = data_dict['contact_time'].copy()
    outc_times = np.round(outc_times * raw.info['sfreq']).astype(int)
    trigger_events = np.hstack((trig_times, np.zeros((len(trig_times), 1), dtype=int),
                                np.ones((len(trig_times), 1), dtype=int)))
    # Epochs aligned on trigger
    trig_epochs = mne.Epochs(raw, trigger_events, tmin=t_times[0], tmax=t_times[1], baseline=t_bline, preload=True)
    trig_epochs.drop_bad()

    # Calculate time series
    tr_times = raw.times[trig_times]
    ac_times = raw.times[acti_times]
    ou_times = raw.times[outc_times]
    reac_times = ac_times - tr_times
    cont_times = ou_times - tr_times
    avg_rt = np.average(reac_times)
    avg_cn = np.average(cont_times)

    # Lowering the sampling frequency to 1KHz
    trig_epochs.resample(1000)
    epochs_tfr = trig_epochs.copy()
    # Filtering in a range
    # trig_epochs.filter(70, 150)

    # trig_epochs.drop([9,12,13,14,15,16])

    # lfp epochs
    fig0 = plt.figure(0)
    trig_epochs.plot_image([0], fig=fig0, show=False)
    ax00, ax01, ax02 = fig0.axes[0], fig0.axes[1], fig0.axes[2]
    ax01.axvline(-1.5, color='y')
    ax01.axvline(-1., color='y')
    ax01.axvline(avg_rt, color='r')
    ax01.axvline(avg_cn, color='k', linestyle='--')
    fig0.savefig(plot_dir.format(subject,condition) + trial_num + os.sep + 'epochs_lfp')
    # mua epochs
    fig1 = plt.figure(1)
    trig_epochs.plot_image([1], fig=fig1, show=False)
    ax10, ax11, ax12 = fig1.axes[0], fig1.axes[1], fig1.axes[2]
    ax11.axvline(-1.5, color='y')
    ax11.axvline(-1., color='y')
    ax11.axvline(avg_rt, color='r')
    ax11.axvline(avg_cn, color='k', linestyle='--')
    fig1.savefig(plot_dir.format(subject,condition) + trial_num + os.sep + 'epochs_mua')

    # Calculate tfr and itc
    freqs = np.arange(5.0, 185.0, 10.0)
    n_cycles = freqs / 5
    tfr, itc = mne.time_frequency.tfr_morlet(epochs_tfr, freqs, n_cycles, return_itc=True, n_jobs=-1, average=True)

    tmin, tmax = np.round(tfr.times[0], decimals=1) + 0.2, np.round(tfr.times[-1], decimals=1) - 0.2
    # fig2, fig3, fig4, fig5 = plt.figure(2), plt.figure(3), plt.figure(4), plt.figure(5)
    show = False
    fig2 = tfr.plot([0], baseline=t_bline, mode='zlogratio', tmin=tmin, tmax=tmax, vmin=-5, vmax=5, show=show)
    fig3 = tfr.plot([1], baseline=t_bline, mode='zlogratio', tmin=tmin, tmax=tmax, vmin=-5, vmax=5, show=show)
    fig4 = itc.plot([0], baseline=t_bline, tmin=tmin, tmax=tmax, show=show)
    fig5 = itc.plot([1], baseline=t_bline, tmin=tmin, tmax=tmax, show=show)

    for f, fn in zip([fig2, fig3, fig4, fig5],['tfr_lfp', 'tfr_mua', 'itc_lfp', 'itc_mua']):
        f.savefig(plot_dir.format(subject,condition) + trial_num + os.sep + fn)

def area_preprocessing(subject, condition, session, area, pick=['lfp']):
    # Deleting bad sessions based on area
    motor = ['fneu0931', 'fneu1131', 'fneu1277']
    associative = []
    limbic = ['fneu1314']
    for r in limbic:
        if r in session:
            session.remove(r)

    # Infos for calculating epochs / TFR
    t_times = [-2.5, 1.5]
    t_bline = (-2.0, -1.5)
    events_struct = {'trigger': ([-0.5, 0.5], 'trigger'), 'cue': ([-2.0, -1.0], 'trigger')}
    freqs = np.arange(5.0, 185.0, 10.0)
    n_cycles = freqs / 5

    good_trials = []
    for ses in session:
        trial_n = check_area(subject, condition, ses, area)
        if isinstance(trial_n, str):
            good_trials.append(trial_n)

    all_epochs, all_tfr, all_itc = [], [], []
    all_evk_avg, all_evk_sem = {k: [] for k in events_struct.keys()}, {k: [] for k in events_struct.keys()}
    all_avg_rt, all_avg_cn = [], []
    for t in good_trials:
        raw_fname = raw_dir.format(subject, condition) + '{0}_raw.fif'.format(t)
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        raw = raw.pick_channels(pick)
        if not 16666.0 < raw.info['sfreq'] <= 16666.7: continue

        data_dict = read_matfile(subject, condition, 'fneu{0}'.format(t))
        trig_times = data_dict['trigger_time'].copy()
        trig_times = np.round(trig_times * raw.info['sfreq']).astype(int)
        acti_times = data_dict['action_time'].copy()
        acti_times = np.round(acti_times * raw.info['sfreq']).astype(int)
        outc_times = data_dict['contact_time'].copy()
        outc_times = np.round(outc_times * raw.info['sfreq']).astype(int)
        trigger_events = np.hstack((trig_times, np.zeros((len(trig_times), 1), dtype=int),
                                    np.ones((len(trig_times), 1), dtype=int)))
        # Epochs aligned on trigger
        trig_epochs = mne.Epochs(raw, trigger_events, tmin=t_times[0], tmax=t_times[1], baseline=t_bline, preload=True)
        trig_epochs.drop_bad()

        # Calculate time series
        tr_times = raw.times[trig_times]
        ac_times = raw.times[acti_times]
        ou_times = raw.times[outc_times]
        reac_times = ac_times - tr_times
        cont_times = ou_times - tr_times
        avg_rt = np.average(reac_times)
        avg_cn = np.average(cont_times)

        # Lowering the sampling frequency to 1KHz
        trig_epochs.resample(1000)

        # Calculating evoked average  / SEM and zscore
        for k in events_struct.keys():
            event_evk = trig_epochs.copy().crop(events_struct[k][0][0], events_struct[k][0][1])
            evk_avg = event_evk.copy().average()
            evk_sem = event_evk.copy().standard_error()
            all_evk_avg[k].append(evk_avg)
            all_evk_sem[k].append(evk_sem)

        epochs_tfr = trig_epochs.copy()
        tfr, itc = mne.time_frequency.tfr_morlet(epochs_tfr, freqs, n_cycles, return_itc=True, n_jobs=-1, average=True)

        all_epochs.append(trig_epochs)
        all_tfr.append(tfr)
        all_itc.append(itc)

        all_avg_rt.append(avg_rt)
        all_avg_cn.append(avg_cn)

    all_epochs = mne.concatenate_epochs(all_epochs)
    for k in events_struct.keys():
        all_evk_avg[k] = mne.grand_average(all_evk_avg[k])
        all_evk_sem[k] = mne.grand_average(all_evk_sem[k])
    all_tfr = mne.grand_average(all_tfr)
    all_itc = mne.grand_average(all_itc)

    fig0 = plt.figure(0)
    all_epochs.plot_image([0], fig=fig0, show=False)
    ax00, ax01, ax02 = fig0.axes[0], fig0.axes[1], fig0.axes[2]
    ax01.axvline(-1.5, color='y')
    ax01.axvline(-1., color='y')
    ax01.axvline(np.average(all_avg_rt), color='r')
    ax01.axvline(np.average(all_avg_cn), color='k', linestyle='--')
    # fig0.savefig(plot_dir.format(subject,condition) + trial_num + os.sep + 'epochs_lfp')
    plt.show()

    fig1 = plt.figure(1)
    for k in events_struct.keys():
        times = all_evk_avg[k].times
        times -= np.average(times)
        average = all_evk_avg[k].data.squeeze() * 1000
        error = all_evk_sem[k].data.squeeze() * 1000
        plt.plot(times, average, label=k)
        plt.fill_between(times, average-error, average+error, alpha=0.2)
        plt.axvline(np.average(times), color='k', linestyle=':')
        plt.axhline(0, color='k')
        plt.title('Average evoked {0}'.format(area), fontsize=15)
        plt.xlabel('Time')
        plt.ylabel('mV')
        plt.legend()
        plt.tight_layout()
    plt.show()

    tmin, tmax = np.round(all_tfr.times[0], decimals=1) + 0.2, np.round(all_tfr.times[-1], decimals=1) - 0.2
    # fig2, fig3, fig4, fig5 = plt.figure(2), plt.figure(3), plt.figure(4), plt.figure(5)
    show = False
    fig2 = all_tfr.plot([0], baseline=t_bline, mode='zlogratio', tmin=tmin, tmax=tmax, vmin=-10, vmax=10, show=show)
    fig3 = all_itc.plot([0], baseline=t_bline, tmin=tmin, tmax=tmax, show=show)
    plt.show()

    fig4 = plt.figure(4)
    for k in events_struct.keys():
        times = all_evk_avg[k].times
        times -= np.average(times)
        evk_zscore = zscore(all_evk_avg[k].data.squeeze())
        plt.plot(times, evk_zscore, label=k)
        plt.axvline(np.average(times), color='k', linestyle=':')
        plt.axhline(0, color='k')
    plt.title('z-scores evoked {0}'.format(area), fontsize=15)
    plt.xlabel('Time')
    plt.ylabel('zscore(V)')
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    preprocessing('freddie', 'easy', 'fneu0873')