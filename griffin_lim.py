# code adapted from https://github.com/bkvogel/griffin_lim

# Author: Brian K. Vogel
# brian.vogel@gmail.com

from pylab import *
import numpy as np
import audio_utilities
from params import audio_out_path, mel_out_path
from os.path import join

filterbank = audio_utilities.make_mel_filterbank(125, 7600, 80, 513, 22050)


def save_mel_to_wav(mel_spectrogram, filename='out'):
    mel_spectrogram = mel_spectrogram.detach().cpu().numpy()

    inverted_mel_to_linear_freq_spectrogram = np.dot(filterbank.T, mel_spectrogram)
    stft_modified = inverted_mel_to_linear_freq_spectrogram.T
    # Use the Griffin&Lim algorithm to reconstruct an audio signal from the
    # magnitude spectrogram.
    x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim(stft_modified, 1024, 256, 100)

    # The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
    max_sample = np.max(abs(x_reconstruct))
    if max_sample > 1.0:
        x_reconstruct = x_reconstruct / max_sample

    # Save the reconstructed signal to a WAV file.
    audio_utilities.save_audio_to_file(x_reconstruct, 22050, outfile=join(audio_out_path, filename + '.wav'))

    # Save the spectrogram image also.
    clf()
    figure(5)
    imshow(stft_modified.T ** 0.125, origin='lower', cmap=cm.hot, aspect='auto',
           interpolation='nearest')
    colorbar()
    title('Spectrogram used to reconstruct audio')
    xlabel('time index')
    ylabel('frequency bin index')
    savefig(join(mel_out_path, filename + '.png'), dpi=150)