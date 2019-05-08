import random
import os
from os.path import join
from pathlib import Path
import params
import torch
from torch.utils.data import Dataset, Subset
import librosa
from librosa.display import specshow
from unidecode import unidecode
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

character_to_index = {ch: i for i, ch in enumerate(params.all_characters)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def text_to_tensor(label):
    return torch.LongTensor([character_to_index[unidecode(ch)] for ch in label])


def prepare_input(batch):
    batch_list = list(batch)

    batch_list = [x.cuda() for x in batch_list]
    for x in batch_list:
        x.requires_grad = False

    return batch_list


def save_mels_to_png(mels, titles, filename, dpi=100):
    num_mels = len(mels)

    plt.figure(figsize=(6.4 * num_mels, 4.8), dpi=dpi)  # TODO: see if you should clear the figures in some way

    for i, mel in enumerate(mels):
        plt.subplot(1, num_mels, i + 1)
        specshow(mel, cmap=plt.cm.viridis)
        plt.title(titles[i])
        plt.xlabel('Frame')
        plt.ylabel('Frequency Bin Index')
        plt.colorbar()
        # TODO: make heatmap values uniform

    plt.savefig(join(params.mel_out_path, filename + '.png'), bbox_inches='tight', dpi=128)


def load_from_files(filelist_dir, dataset):
    train_stems = parse_filelist(join(filelist_dir, 'train_filelist.txt'))
    val_stems = parse_filelist(join(filelist_dir, 'val_filelist.txt'))
    test_stems = parse_filelist(join(filelist_dir, 'test_filelist.txt'))

    train_idxs = [dataset.stem_to_idx[s] for s in train_stems]
    val_idxs = [dataset.stem_to_idx[s] for s in val_stems]
    test_idxs = [dataset.stem_to_idx[s] for s in test_stems]

    return Subset(dataset, train_idxs), Subset(dataset, val_idxs), Subset(dataset, test_idxs)


def parse_filelist(filename):
    stems = []
    with open(filename, encoding='utf-8', errors='ignore') as filelist:
        for line in filelist:
            path, _ = line.split('|')
            stems.append(Path(path).stem)
    return stems


def dynamic_range_compression(mel, compression_factor=1):
    return np.log(np.clip(mel, a_min=0.01, a_max=None) * compression_factor)


def dynamic_range_decompression(mel, compression_factor=1):
    return torch.exp(mel) / compression_factor


class LabelledMelDataset(Dataset):
    def __init__(self, data_dir, loader_fn):
        self.mel_filterbank = librosa.filters.mel(sr=params.sampling_rate, n_fft=params.n_fft,
                                                  n_mels=params.n_mel_channels, fmin=params.f_min, fmax=params.f_max)

        self.paths, self.labels = loader_fn(data_dir)

        self.stem_to_idx = {p.stem: i for i, p in enumerate(self.paths)}

    def label_to_list(self, label):
        return [character_to_index[unidecode(ch)] for ch in label]

    def audiopath_to_mel(self, path):
        y, sr = librosa.load(path.as_posix())

        # ensure sampling rate is the same
        assert(sr == params.sampling_rate)

        # audio already normalised but just in case
        assert(np.min(y) >= -1)
        assert(np.max(y) <= 1)

        stft = librosa.stft(y, n_fft=params.n_fft)
        stft_mag = np.absolute(stft)
        mel = np.dot(self.mel_filterbank, stft_mag)

        mel = dynamic_range_compression(mel)

        return mel

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
            torch.stack(padded_mels).transpose(1, 2), torch.LongTensor(mel_lengths),\
            torch.stack(padded_stop_tokens)


class LJSpeechLoader:
    def __init__(self):
        pass

    def __call__(self, data_dir):
        data_path = Path(data_dir)

        paths, labels = [], []
        with open(os.path.join(data_dir, "metadata.csv"), encoding='utf-8') as metadata:
            for line in metadata:
                path, _, label = line.split('|')
                paths.append(data_path / 'wavs' / (path + '.wav'))
                labels.append(label)

        return paths, labels
