import numpy as np
from scipy.fftpack import fft, fftfreq


def fourier_transform(dat: np.ndarray, window: int):
    n, d = dat.shape

    groups = dict()
    for i in range(d):
        u_dat = dat[:, i]
        fft_series = fft(u_dat)
        power = np.abs(fft_series)
        sample_freq = fftfreq(fft_series.size)
        
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        powers = power[pos_mask]

        top_k_seasons = 1
        # top K=3 index
        top_k_idxs = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
        fft_periods = (1 / freqs[top_k_idxs]).astype(int)

        if fft_periods[0] < 12 or fft_periods[0] > (n // 2):
        # if fft_periods[0] < 12 or fft_periods[0] > 288:
            if 'non-periodic' in groups:
                groups['non-periodic'].append(i)
            else:
                groups['non-periodic'] = [i]
        else:
            if fft_periods[0] in groups:
                groups[fft_periods[0]].append(i)
            else:
                groups[fft_periods[0]] = [i]

    return groups
