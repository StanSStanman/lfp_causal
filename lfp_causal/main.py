import os
import mne
import matplotlib.pyplot as plt
from lfp_causal.epochs import (create_epochs,
                               visualize_epochs,
                               reject_bad_epochs,
                               auto_drop_chans)
from lfp_causal.tf_analysis import (epochs_tf_analysis,
                                    evoked_tf_analysis)
from lfp_causal.filters import convolfil


if __name__ == '__main__':

    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_on'

    freqs = (15, 70)

    show = False
    if show is False:
        import matplotlib
        matplotlib.use('Agg')

    raw_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/raw'.format(monkey, condition)
    eve_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/eve'.format(monkey, condition)
    epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)
    rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
               '{0}/{1}/recording_info.xlsx'.format(monkey, condition)
    fig_dir = '/home/jerry/Scrivania/figures/lfp' \
              '/{0}/{1}'.format(monkey, condition)

    files = []
    for file in os.listdir(raw_dir):
        # file = '1398_raw.fif'
        if file.endswith('.fif'):
            session = file.replace('_raw.fif', '')
            fname_raw = os.path.join(raw_dir, file)
            fname_eve = os.path.join(eve_dir, file.replace('raw', 'eve'))
            fname_epo = os.path.join(epo_dir,
                                     file.replace('raw',
                                                  '{0}_epo'.format(event)))
            if not os.path.exists(os.path.join(fig_dir, session)):
                os.mkdir(os.path.join(fig_dir, session))

            # create_epochs(fname_raw, fname_eve,
            #               event, -2.5, 1.,
            #               (-2.5, -1.), fname_epo,
            #               ch_drop='manual')

            bad_ch = auto_drop_chans(rec_info, session)

            epo = create_epochs(fname_raw, fname_eve,
                                event, -2., 1.5,
                                None, fname_epo,
                                ch_drop=bad_ch)
            # create_epochs(fname_raw, fname_eve,
            #               event, -2.5, 1.,
            #               (-2., -1.), fname_epo,
            #               ch_drop=bad_ch)

            epo = reject_bad_epochs(fname_epo, monkey, condition, session)[0]
            # epo = convolfil(epo, 100, 'hann')

            epo.save(epo.filename, overwrite=True)
            epo = mne.read_epochs(fname_epo)

            epo_plot = visualize_epochs(fname_epo, show=show)
            epo_plot.savefig(os.path.join(fig_dir, session, 'epochs'), dpi=300)

            _, _, (ep_pow, ep_itc) = epochs_tf_analysis(fname_epo,
                                                        freqs=freqs,
                                                        baseline=(.2, .7),
                                                        avg=True,
                                                        show=show)
            ep_pow.savefig(os.path.join(fig_dir, session, 'epo_pow'), dpi=300)
            ep_itc.savefig(os.path.join(fig_dir, session, 'epo_itc'), dpi=300)

            evo = epo.average()
            evo_plot = evo.plot(show=show)
            evo_plot.savefig(os.path.join(fig_dir, session, 'evoked'), dpi=300)

            _, (ev_pow, _) = evoked_tf_analysis(evo, freqs=freqs,
                                                baseline=(.2, .7),
                                                show=show)
            ev_pow.savefig(os.path.join(fig_dir, session, 'evo_tfr'), dpi=300)

            plt.close('all')

            # visualize_epochs(fname_epo, ['LFP1'])
            # visualize_epochs(fname_epo, ['LFP2'])
            # epochs_tf_analysis(fname_epo, baseline=(-2.8, -2.6), avg=True)
            # epochs_tf_analysis(fname_epo, baseline=(-2.8, -2.6), avg=False)
