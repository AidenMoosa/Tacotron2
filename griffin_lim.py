# code adapted from https://github.com/bkvogel/griffin_lim

# Author: Brian K. Vogel
# brian.vogel@gmail.com

import audio_utilities
from params import audio_out_path, mel_out_path
from os.path import join
import librosa
import params
import numpy as np

filterbank = librosa.filters.mel(sr=params.sampling_rate, n_fft=params.n_fft, n_mels=params.n_mel_channels,
                                 fmin=params.f_min, fmax=params.f_max)


def save_mel_to_wav(mel_spectrogram, filename='out'):
    inverted_mel_to_linear_freq_spectrogram = np.dot(filterbank.T, mel_spectrogram)
    stft_modified = inverted_mel_to_linear_freq_spectrogram.T

    # Clip values to avoid artifacts
    # stft_modified[stft_modified > 0.05] = 0
    # stft_modified = np.clip(stft_modified, a_min=None, a_max=0.05)

    # Use the Griffin&Lim algorithm to reconstruct an audio signal from the
    # magnitude spectrogram.
    x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim(stft_modified, params.n_fft, params.frame_hop,
                                                                   params.griffin_lim_iters)

    # The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
    max_sample = np.max(abs(x_reconstruct))
    if max_sample > 1.0:
        x_reconstruct = x_reconstruct / max_sample

    # Save the reconstructed signal to a WAV file.
    audio_utilities.save_audio_to_file(x_reconstruct, params.sampling_rate, outfile=join(audio_out_path, filename + '.wav'))

    # Return mel spectrogram data
    return stft_modified.T
