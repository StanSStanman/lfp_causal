import mne
import numpy as np


def single_channel_ica(epochs, ch_names=None, new_chans=False, save=False):
    if isinstance(epochs, str):
        epochs = mne.read_epochs(epochs, preload=True)
    epochs.load_data()

    if ch_names is None:
        ch_names = epochs.ch_names
    for ch in ch_names:
        epo = epochs.copy().pick([ch])

        fake_ch = ['{0}_{1}'.format(ch, x)
                   for x in range(epo.get_data().shape[0])]
        info_ica = mne.create_info(fake_ch, sfreq=epo.info['sfreq'],
                               ch_types='seeg')
        trans_epo = np.expand_dims(epo.get_data().squeeze(), 0)

        epo_ica = mne.EpochsArray(trans_epo, info=info_ica,
                                  tmin=epo.tmin)

        ica = mne.preprocessing.ICA(n_components=0.95,
                                    random_state=23,
                                    method='fastica')
        ica.fit(epo_ica)
        ica.plot_sources(epo_ica, stop=2, show=True, block=True)
        ica.apply(epo_ica)

        trans_ica = np.expand_dims(epo_ica.get_data().squeeze(), 1)
        epo._data = trans_ica

        if new_chans:
            epo.rename_channels({ch: ch + '_ICA'})
            if epo.ch_names[0] in epochs.ch_names:
                epochs.drop_channels([ch + '_ICA'])
            epochs.add_channels([epo], force_update_info=True)
            # To solve a bug in mne
            # TODO: open an issue on mne
            epochs.picks = np.array(range(len(epochs.ch_names)))
        elif not new_chans:
            epochs.rename_channels({ch: ch + '_DEL'})
            epochs.add_channels([epo], force_update_info=True)
            epochs.picks = np.array(range(len(epochs.ch_names)))
            epochs.drop_channels([ch + '_DEL'])
            # To solve a bug in mne
            # TODO: open an issue on mne
            epochs.picks = np.array(range(len(epochs.ch_names)))

    if not new_chans:
        epochs.reorder_channels(ch_names)

    if isinstance(save, str):
        epochs.save(save, overwrite=True)
    elif isinstance(save, bool):
        if save:
            epochs.save(epochs.filename, overwrite=True)

    return epochs


if __name__ == '__main__':
    import os
    from lfp_causal.epochs import (visualize_epochs,
                                   concatenate_epochs,
                                   reject_bad_epochs)
    from lfp_causal.tf_analysis import (epochs_tf_analysis,
                                        tf_diff)

    monkey = 'freddie'
    condition = 'easy'
    event = 'mov_on'

    epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)

    files = []
    for file in os.listdir(epo_dir):
        rej = [] #['1024', '1036', '1038', '1043', '1699', '1215',
               #'0981', '1165', '1204', '1209', '1255']
        file = '0975_{0}_epo.fif'.format(event)
        if file.endswith('{0}_epo.fif'.format(event)) and \
                file.split(sep='_')[0] not in rej:
            epo_fname = os.path.join(epo_dir, file)

    # Concatenate epochs and perform ICA
            session = file.split(sep='_')[0]
            epo = reject_bad_epochs(epo_fname, monkey, condition, session)[0]
            epo.rename_channels({epo.ch_names[0]: 'LFP'})
            epo.apply_baseline((-2.3, -.3))
            files.append(epo)
    all_epo = mne.concatenate_epochs(files)
    all_epo.plot_psd(fmin=15., fmax=140.)
    # visualize_epochs(all_epo, block=False)
    # all_epo_ica = single_channel_ica(all_epo.copy(), new_chans=False,
    #                              save=False)
    # visualize_epochs(all_epo_ica)
    pow_epo, itc_epo = epochs_tf_analysis(all_epo,
                                          baseline=(-2.5, -.5),
                                          freqs=np.logspace(
                                              *np.log10([15, 80]), num=100),
                                          avg=True, show=True)
    # pow_ica, itc_ica = epochs_tf_analysis(all_epo_ica,
    #                                       baseline=(-2., -1.),
    #                                       avg=True)

    # tf_diff(pow_epo, pow_ica, baseline=None, tmin=-.3, tmax=.5)



            # # Single channel ICA
            # session = file.split(sep='_')[0]
            # epochs = reject_bad_epochs(epo_fname, monkey, condition, session)
            # for epo in epochs:
            #     epo.apply_baseline((-2.3, -.3))
            #     epo.plot_psd(fmin=15., fmax=140.)
            #     # visualize_epochs(epo, block=False)
            #     # # pow_epo, itc_epo = epochs_tf_analysis(epo,
            #     # #                                       baseline=(-2.3, -2.),
            #     # #                                       avg=True, show=False)
            #     # epo_ica = single_channel_ica(epo.copy(), new_chans=False,
            #     #                              save=False)
            #     # visualize_epochs(epo_ica)
            #     pow_epo, itc_epo = epochs_tf_analysis(epo,
            #                                           baseline=(-2.3, -0.3),
            #                                           freqs=np.logspace(*np.log10([15, 80]), num=100),
            #                                           avg=True, show=True)
            #     # pow_ica, itc_ica = epochs_tf_analysis(epo_ica,
            #     #                                       baseline=(-2.3, -2.),
            #     #                                       avg=True)
            #
            #     # tf_diff(pow_epo, pow_ica, baseline=None, tmin=-.3, tmax=.5)
