import logging
import mne
from lfp_causal.directories import *
from lfp_causal.old.controls import quality_check, check_area
from lfp_causal.old.read_infos import files_info
from lfp_causal.frequences_analysis import time_frequency_analysis

subject = 'freddie'
condition = 'easy'
session_list = quality_check(subject, condition, min_quality=1)
events_struct = {'trigger':([-0.5, 0.5], 'trigger'),
                 'cue':([-2.0, -1.0], 'trigger')}

# for session in session_list:
#     if ('fneu' + session) in files_info(subject, condition)[0]:
#         create_rawfiles(subject, condition, session)
#         create_epochs(subject, condition, session)


associative_fname, limbic_fname, motor_fname = [], [], []
for session in session_list:
    if ('fneu' + session) in files_info(subject, condition)[0]:
        associative_fname.append(check_area(subject,
                                            condition,
                                            session,
                                            'associative striatum'))
        limbic_fname.append(check_area(subject,
                                       condition,
                                       session,
                                       'limbic striatum'))
        motor_fname.append(check_area(subject,
                                      condition,
                                      session,
                                      'motor striatum'))
    else:
        logging.warning(
            'session {0} is not available from current data'.format(session))
associative_fname = list(filter(lambda x: x != None, associative_fname))
limbic_fname = list(filter(lambda x: x != None, limbic_fname))
motor_fname = list(filter(lambda x: x != None, motor_fname))

_epochs = []
for session in motor_fname:
    epo = mne.read_epochs(epochs_dir.format(subject,
                                            condition,
                                            session) +
                          '{0}_{1}-epo.fif'.format(session, 'trigger'))

    epo = epo.pick_channels(['lfp'])
    epo._data = epo._data/epo._data.std(0)

    _epochs.append(epo)
epochs = mne.concatenate_epochs(_epochs)
epochs.load_data()
epochs.resample(2000, n_jobs=-1)
# epochs.close()
epochs.plot_psd(fmax=200, picks=[0])
epochs.plot_image([0])

tfr = time_frequency_analysis(epochs=epochs, band='hga')

# freqs = np.logspace(5.6, 8., 12, base=2)
# n_cycles = 16
# t_bd = 12
# tfr = mne.time_frequency.tfr_multitaper(epochs, freqs, n_cycles, time_bandwidth=t_bd, return_itc=False, n_jobs=-1, average=False)
#
# for e in range(tfr.data.shape[0]):
#     for c in range(tfr.data.shape[1]):
#         for f in range(tfr.data.shape[2]):
#             tfr.data[e, c, f, :] = zscore(tfr.data[e, c, f, :])
#
# plt.plot(btfr.times, btfr.data[:,0,5,:].squeeze().mean(0).T)

# plt.pcolormesh(b_tfr_lfp.data.squeeze(), cmap='jet', vmin=-20, vmax=20)
# plt.colorbar()
# plt.vlines(np.where(tfr.times == 0)[0], 0, 589)