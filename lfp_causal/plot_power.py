import xarray as xr
import os
import os.path as op
import matplotlib.pyplot as plt
from lfp_causal.compute_power import normalize_power

folder = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/freddie/easy/pow/0990'

data = xr.load_dataset(op.join(folder, 'trig_off_pow_5_120.nc'))

norm_data = normalize_power(data, 'relchange', (-1.8, -1.3))
norm_data_cut = norm_data.loc[dict(times=slice(-.2, .8))]
norm_data_cut = norm_data_cut.loc[dict(freqs=slice(50, 120))]
plt.pcolormesh(norm_data_cut.times, norm_data_cut.freqs, norm_data_cut.to_array().squeeze().mean('trials'))
plt.colorbar()
plt.show()

plt.plot(norm_data.loc[dict(freqs=slice(5, 15))].to_array().squeeze().mean('times').mean('freqs'), label='5-15Hz')
plt.plot(norm_data.loc[dict(freqs=slice(15, 30))].to_array().squeeze().mean('times').mean('freqs'), label='15-30Hz')
plt.plot(norm_data.loc[dict(freqs=slice(30, 50))].to_array().squeeze().mean('times').mean('freqs'), label='30-50Hz')
plt.plot(norm_data.loc[dict(freqs=slice(50, 80))].to_array().squeeze().mean('times').mean('freqs'), label='50-80Hz')
plt.plot(norm_data.loc[dict(freqs=slice(80, 120))].to_array().squeeze().mean('times').mean('freqs'), label='80-120Hz')
plt.legend()
plt.show()

plt.pcolormesh(norm_data.loc[dict(freqs=slice(5, 15))].to_array().squeeze().mean('freqs'), label='5-15Hz')
plt.pcolormesh(norm_data.loc[dict(freqs=slice(15, 30))].to_array().squeeze().mean('freqs'), label='15-30Hz')
plt.pcolormesh(norm_data.loc[dict(freqs=slice(30, 50))].to_array().squeeze().mean('freqs'), label='30-50Hz')
plt.pcolormesh(norm_data.loc[dict(freqs=slice(50, 80))].to_array().squeeze().mean('freqs'), label='50-80Hz')
plt.pcolormesh(norm_data.loc[dict(freqs=slice(80, 120))].to_array().squeeze().mean('freqs'), label='80-120Hz')
