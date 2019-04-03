# code taken from https://github.com/bkvogel/griffin_lim

# Author: Brian K. Vogel
# brian.vogel@gmail.com

import wave
import scipy
import scipy.signal
import array
import scipy.io.wavfile
import torch
import numpy as np


def hz_to_mel(f_hz):
    """Convert Hz to mel scale.

    This uses the formula from O'Shaugnessy's book.
    Args:
        f_hz (float): The value in Hz.

    Returns:
        The value in mels.
    """
    return 2595*np.log10(1.0 + f_hz/700.0)


def mel_to_hz(m_mel):
    """Convert mel scale to Hz.

    This uses the formula from O'Shaugnessy's book.
    Args:
        m_mel (float): The value in mels

    Returns:
        The value in Hz
    """
    return 700*(10**(m_mel/2595) - 1.0)


def fft_bin_to_hz(n_bin, sample_rate_hz, fft_size):
    """Convert FFT bin index to frequency in Hz.

    Args:
        n_bin (int or float): The FFT bin index.
        sample_rate_hz (int or float): The sample rate in Hz.
        fft_size (int or float): The FFT size.

    Returns:
        The value in Hz.
    """
    n_bin = float(n_bin)
    sample_rate_hz = float(sample_rate_hz)
    fft_size = float(fft_size)
    return n_bin*sample_rate_hz/(2.0*fft_size)


def hz_to_fft_bin(f_hz, sample_rate_hz, fft_size):
    """Convert frequency in Hz to FFT bin index.

    Args:
        f_hz (int or float): The frequency in Hz.
        sample_rate_hz (int or float): The sample rate in Hz.
        fft_size (int or float): The FFT size.

    Returns:
        The FFT bin index as an int.
    """
    f_hz = float(f_hz)
    sample_rate_hz = float(sample_rate_hz)
    fft_size = float(fft_size)
    fft_bin = int(np.round((f_hz*2.0*fft_size/sample_rate_hz)))
    if fft_bin >= fft_size:
        fft_bin = fft_size-1
    return fft_bin


def stft_for_reconstruction(x, fft_size, hopsamp):
    """Compute and return the STFT of the supplied time domain signal x.

    Args:
        x (1-dim Numpy array): A time domain signal.
        fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
        hopsamp (int):

    Returns:
        The STFT. The rows are the time slices and columns are the frequency bins.
    """
    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    return np.array([np.fft.rfft(window*x[i:i+fft_size])
                     for i in range(0, len(x)-fft_size, hopsamp)])


def istft_for_reconstruction(X, fft_size, hopsamp):
    """Invert a STFT into a time domain signal.

    Args:
        X (2-dim Numpy array): Input spectrogram. The rows are the time slices and columns are the frequency bins.
        fft_size (int):
        hopsamp (int): The hop size, in samples.

    Returns:
        The inverse STFT.
    """
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n, i in enumerate(range(0, len(x)-fft_size, hopsamp)):
        x[i:i+fft_size] += window*np.real(np.fft.irfft(X[n]))
    return x


def get_signal(in_file, expected_fs=44100):
    """Load a wav file.

    If the file contains more than one channel, return a mono file by taking
    the mean of all channels.

    If the sample rate differs from the expected sample rate (default is 44100 Hz),
    raise an exception.

    Args:
        in_file: The input wav file, which should have a sample rate of `expected_fs`.
        expected_fs (int): The expected sample rate of the input wav file.

    Returns:
        The audio siganl as a 1-dim Numpy array. The values will be in the range [-1.0, 1.0]. fixme ( not yet)
    """
    fs, y = scipy.io.wavfile.read(in_file)
    num_type = y[0].dtype
    if num_type == 'int16':
        y = y*(1.0/32768)
    elif num_type == 'int32':
        y = y*(1.0/2147483648)
    elif num_type == 'float32':
        # Nothing to do
        pass
    elif num_type == 'uint8':
        raise Exception('8-bit PCM is not supported.')
    else:
        raise Exception('Unknown format.')
    if fs != expected_fs:
        raise Exception('Invalid sample rate.')
    if y.ndim == 1:
        return y
    else:
        return y.mean(axis=1)


def reconstruct_signal_griffin_lim(magnitude_spectrogram, fft_size, hopsamp, iterations):
    """Reconstruct an audio signal from a magnitude spectrogram.

    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.

    Args:
        magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
            and the columns correspond to frequency bins.
        fft_size (int): The FFT size, which should be a power of 2.
        hopsamp (int): The hope size in samples.
        iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
            is sufficient.

    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    time_slices = magnitude_spectrogram.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(len_samples)
    n = iterations  # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct, fft_size, hopsamp)
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        # prev_x = x_reconstruct
        x_reconstruct = istft_for_reconstruction(proposal_spectrogram, fft_size, hopsamp)
        # diff = np.sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        # print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct


def save_audio_to_file(x, sample_rate, outfile='out.wav'):
    """Save a mono signal to a file.

    Args:
        x (1-dim Numpy array): The audio signal to save. The signal values should be in the range [-1.0, 1.0].
        sample_rate (int): The sample rate of the signal, in Hz.
        outfile: Name of the file to save.

    """
    x_max = np.max(abs(x))
    assert x_max <= 1.0, 'Input audio value is out of range. Should be in the range [-1.0, 1.0].'
    x /= x_max
    x *= 32767.0

    data = array.array('h')
    for i in range(len(x)):
        cur_samp = int(round(x[i]))
        data.append(cur_samp)

    f = wave.open(outfile, 'w')
    f.setparams((1, 2, sample_rate, 0, "NONE", "Uncompressed"))
    f.writeframes(data.tostring())
    f.close()


def dynamic_range_compression(mel, compression_factor=100):
    return np.log(np.clip(mel, a_min=0.01, a_max=None) * compression_factor)


def dynamic_range_decompression(mel, compression_factor=100):
    return np.exp(mel) / compression_factor


# from https://github.com/pytorch/audio/blob/master/torchaudio/transforms.py
class MuLawEncoding(object):
    """Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_
    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1
    Args:
        quantization_channels (int): Number of channels. default: 256
    """

    def __init__(self, quantization_channels=256):
        self.qc = quantization_channels

    def __call__(self, x):
        """
        Args:
            x (FloatTensor/LongTensor or ndarray)
        Returns:
            x_mu (LongTensor or ndarray)
        """
        mu = self.qc - 1.
        if isinstance(x, np.ndarray):
            x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).astype(int)
        elif isinstance(x, torch.Tensor):
            if not x.dtype.is_floating_point:
                x = x.to(torch.float)
            mu = torch.tensor(mu, dtype=x.dtype)
            x_mu = torch.sign(x) * torch.log1p(mu *
                                               torch.abs(x)) / torch.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()
        return x_mu

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MuLawExpanding(object):
    """Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_
    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.
    Args:
        quantization_channels (int): Number of channels. default: 256
    """

    def __init__(self, quantization_channels=256):
        self.qc = quantization_channels

    def __call__(self, x_mu):
        """
        Args:
            x_mu (FloatTensor/LongTensor or ndarray)
        Returns:
            x (FloatTensor or ndarray)
        """
        mu = self.qc - 1.
        if isinstance(x_mu, np.ndarray):
            x = (x_mu / mu) * 2 - 1.
            x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
        elif isinstance(x_mu, torch.Tensor):
            if not x_mu.dtype.is_floating_point:
                x_mu = x_mu.to(torch.float)
            mu = torch.tensor(mu, dtype=x_mu.dtype)
            x = (x_mu / mu) * 2 - 1.
            x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'
