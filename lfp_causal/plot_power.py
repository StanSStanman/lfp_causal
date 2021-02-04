import xarray as xr
import os
import os.path as op
import matplotlib.pyplot as plt
from research.get_dirs import get_dirs
from lfp_causal.compute_power import normalize_power
from lfp_causal.IO import read_sector
from lfp_causal.profiling import (RepeatedTimer, memory_usage, cpu_usage)


def plot_avg_tf(powers, blines=None):
    avg_pow = None
    if blines is None:
        blines = range(avg_pow)
    for p, b in zip(powers, blines):
        print('Loading', p)
        pow = xr.load_dataset(p)
        if isinstance(b, (int, float)):
            pow = normalize_power(pow, 'relchange', (-1.8, -1.3))
        elif isinstance(b, str):
            pow = normalize_power(pow, 'fbline_tt_zs', (-.51, -.01), file=b)
        pow = pow.loc[dict(times=slice(-.15, .8))]
        # pow = pow.loc[dict(freqs=slice(70, 120))]
        pow = pow.to_array().squeeze().mean('trials')

        if avg_pow is None:
            avg_pow = pow
        else:
            avg_pow = (avg_pow + pow) / 2
    fig, ax = plt.subplots(1, 1)
    cm = ax.pcolormesh(avg_pow.times, avg_pow.freqs, avg_pow)
    plt.colorbar(cm)
    plt.show()
    return


if __name__ == '__main__':
    monkey = 'freddie'
    condition = 'hard'
    file = 'trig_off_pow_8_120_sl.nc'
    bline = 'cue_on_pow_8_120_sl.nc'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    # sectors = ['motor striatum', 'limbic striatum']

    dirs = get_dirs('local', 'lfp_causal')
    directory = dirs['pow'].format(monkey, condition)
    rec_info = op.join(dirs['ep_cnds'].format(monkey, condition),
                       'files_info.xlsx')

    rej_files = ['0845', '0847', '0873', '0939', '0945', '1038', '1204',
                 '1217'] + \
                ['0944', '0967', '0969', '0967', '0970', '0971', '1139',
                 '1145', '1515', '1701']

    for sect in sectors:
        fid = read_sector(rec_info, sect)

        all_files = []
        all_bline = []
        for fs in fid['file'][:5]:
            fname = op.join(directory, fs, file)
            bname = op.join(directory, fs, bline)
            if op.exists(fname) and fs not in rej_files:
                all_files.append(fname)
                all_bline.append(bname)

        mu = RepeatedTimer(1, memory_usage)
        cu = RepeatedTimer(1, cpu_usage)
        import time
        time.sleep(5)
        plot_avg_tf(all_files, all_bline)
        mu.stop()
        cu.stop()
        print(cpu_usage())

        print('done')



# folder = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/freddie/easy/pow/0990'
#
# data = xr.load_dataset(op.join(folder, 'trig_off_pow_8_120_sl.nc'))
#
# norm_data = normalize_power(data, 'relchange', (-1.8, -1.3))
# norm_data_cut = norm_data.loc[dict(times=slice(-.2, .8))]
# norm_data_cut = norm_data_cut.loc[dict(freqs=slice(50, 120))]
# plt.pcolormesh(norm_data_cut.times, norm_data_cut.freqs, norm_data_cut.to_array().squeeze().mean('trials'))
# plt.colorbar()
# plt.show()
#
# plt.plot(norm_data.loc[dict(freqs=slice(5, 15))].to_array().squeeze().mean('times').mean('freqs'), label='5-15Hz')
# plt.plot(norm_data.loc[dict(freqs=slice(15, 30))].to_array().squeeze().mean('times').mean('freqs'), label='15-30Hz')
# plt.plot(norm_data.loc[dict(freqs=slice(30, 50))].to_array().squeeze().mean('times').mean('freqs'), label='30-50Hz')
# plt.plot(norm_data.loc[dict(freqs=slice(50, 80))].to_array().squeeze().mean('times').mean('freqs'), label='50-80Hz')
# plt.plot(norm_data.loc[dict(freqs=slice(80, 120))].to_array().squeeze().mean('times').mean('freqs'), label='80-120Hz')
# plt.legend()
# plt.show()
#
# plt.pcolormesh(norm_data.loc[dict(freqs=slice(5, 15))].to_array().squeeze().mean('freqs'), label='5-15Hz')
# plt.pcolormesh(norm_data.loc[dict(freqs=slice(15, 30))].to_array().squeeze().mean('freqs'), label='15-30Hz')
# plt.pcolormesh(norm_data.loc[dict(freqs=slice(30, 50))].to_array().squeeze().mean('freqs'), label='30-50Hz')
# plt.pcolormesh(norm_data.loc[dict(freqs=slice(50, 80))].to_array().squeeze().mean('freqs'), label='50-80Hz')
# plt.pcolormesh(norm_data.loc[dict(freqs=slice(80, 120))].to_array().squeeze().mean('freqs'), label='80-120Hz')
