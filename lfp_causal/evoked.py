import mne
import matplotlib.pyplot as plt

def epo_to_evo(epochs):
    if isinstance(epochs, str):
        epochs = mne.read_epochs(epochs)

    evoked = epochs.copy().average()

    return evoked


def plot_evoked(evoked, fig=None, color=None, label=None,
                vh_lines=True, show=True):
    import mplcursors

    if isinstance(evoked, str):
        if evoked.endswith('epo.fif'):
            evoked = mne.read_epochs(evoked)
        elif evoked.endswith('evo.fif'):
            evoked = mne.read_evokeds(evoked)
    if isinstance(evoked, (mne.Epochs, mne.BaseEpochs, mne.EpochsArray)):
        evoked = epo_to_evo(evoked)

    # evoked.crop(-.5, 1.)

    data = evoked.data.squeeze()
    times = evoked.times.squeeze()

    if fig is None:
        fig, ax = plt.subplots(1, 1)
    elif isinstance(fig, plt.Figure):
        ax = fig.axes[0]
    else:
        raise ValueError('Fig must be a figure containing one axes')

    ax.plot(times, data, color=color, label=label)
    if vh_lines == True:
        ax.axvline(0, linestyle='--', color='k', linewidth=.8)
        ax.axhline(0, linestyle='-', color='k', linewidth=.8)

    if show:
        fig.show()
    mplcursors.cursor(hover=True)

    return fig


if __name__ == '__main__':
    import os
    from lfp_causal.IO import read_sector
    # import mplcursors

    monkey = 'freddie'
    condition = 'easy'
    event = 'mov_on'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']

    rec_info = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/' \
               '{0}/{1}/recording_info.xlsx'.format(monkey, condition)
    epo_dir = '/media/jerry/TOSHIBA EXT/data/db_lfp/' \
              'lfp_causal/{0}/{1}/epo'.format(monkey, condition)

    for sect in sectors:
        fid = read_sector(rec_info, sect)

        # figures = []
        fig, ax = plt.subplots(1, 1)
        for file in fid['file']:
            # file = '0539'
            fname_epo = os.path.join(epo_dir,
                                     '{0}_{1}_epo.fif'.format(file, event))

            if os.path.exists(fname_epo):

                fig = plot_evoked(fname_epo, fig=fig, color=None,
                                  label=file, show=False)

        plt.legend()
        # mplcursors.cursor(hover=True)
        plt.show()
