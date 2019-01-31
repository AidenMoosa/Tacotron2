import os
import params
import torch
from torch.utils.data import Dataset
import librosa
from audio_utilities import dynamic_range_compression
import numpy as np
from unidecode import unidecode

character_to_index = {ch: i for i, ch in enumerate(params.all_characters)}


def text_to_tensor(label):
    return torch.LongTensor([character_to_index[unidecode(ch)] for ch in label])


class LabelledMelDataset(Dataset):
    def __init__(self, root_dir, loader_fn):
        self.paths, self.labels = loader_fn(root_dir)
        self.root_dir = root_dir

        self.mel_filterbank = librosa.filters.mel(sr=22050, n_fft=params.n_fft, n_mels=params.n_mel_channels,
                                                  fmin=params.f_min, fmax=params.f_max)

    def label_to_list(self, label):
        return [character_to_index[unidecode(ch)] for ch in label]

    def audiopath_to_mel(self, path):
        y, sr = librosa.load(path)

        # ensure sampling rate is the same
        assert(sr == params.sampling_rate)

        # audio already normalised but just in case
        assert(np.min(y) >= -1)
        assert(np.max(y) <= 1)

        stft = librosa.stft(y, n_fft=params.n_fft)
        stft_mag = np.absolute(stft)
        mel = np.dot(self.mel_filterbank, stft_mag)

        mel = dynamic_range_compression(mel)

        return mel.real

    def __getitem__(self, idx):
        text = self.label_to_list(self.labels[idx])
        mel = self.audiopath_to_mel(self.paths[idx])

        return text, mel

    def __len__(self):
        return len(self.paths)


# ideas taken from https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8
class PadCollate:
    def __init__(self):
        pass

    def pad_tensor(self, tensor, target_len, dim=-1):
        pad_shape = list(tensor.shape)
        pad_shape[dim] = target_len - tensor.size(dim)

        return torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype)], dim=dim)

    def __call__(self, batch):
        # sort batch into descending text length order
        sorted_batch = sorted(batch, reverse=True, key=lambda text_mel: len(text_mel[0]))

        # extract separate lists from (text, mel) pairs
        texts, mels = zip(*sorted_batch)

        text_lengths = [len(text) for text in texts]
        max_text_length = max(text_lengths)
        padded_texts = [self.pad_tensor(torch.LongTensor(text), max_text_length, dim=0) for text in texts]

        mel_lengths = [mel.shape[-1] for mel in mels]
        max_mel_length = max(mel_lengths)
        padded_mels = [self.pad_tensor(torch.from_numpy(mel).float(), max_mel_length) for mel in mels]

        padded_stop_tokens = [mel.new_zeros(max_mel_length) for mel in padded_mels]
        for i, l in enumerate(mel_lengths):
            padded_stop_tokens[i][mel_lengths[i]:] = 1

        return torch.stack(padded_texts), torch.LongTensor(text_lengths), \
            torch.stack(padded_mels), torch.LongTensor(mel_lengths),\
            torch.stack(padded_stop_tokens)


class LibriSpeechLoader:
    def __init__(self):
        pass

    def __call__(self, root_dir):
        paths, labels = [], []
        for root, dirs, files in os.walk(root_dir, topdown=True):
            for path in files:
                if path.endswith('.txt'):
                    with open(os.path.join(root, path)) as file:
                        string = ''
                        in_name = True

                        while True:
                            c = file.read(1)
                            if not c:
                                string = string.strip()
                                labels.append(string)
                                break

                            if in_name:
                                if c.isspace():
                                    in_name = False
                                    string = string.strip()
                                    paths.append(os.path.join(root, string) + ".flac")
                                    string = ''
                            else:
                                if c.isdigit():
                                    in_name = True
                                    string = string.strip()
                                    labels.append(string)
                                    string = ''

                            string += c

        return paths, labels


class LJSpeechLoader:
    def __init__(self):
        pass

    def __call__(self, root_dir):
        paths, labels = [], []
        with open(os.path.join(root_dir, "metadata.csv"), encoding='utf-8') as metadata:
            for line in metadata:
                path, _, label = line.split('|')
                paths.append(os.path.join(root_dir, 'wavs', path + '.wav'))
                labels.append(label)

        return paths, labels
