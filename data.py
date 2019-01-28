import os
from params import all_characters, n_characters
import torch
from torch.utils.data import Dataset
import librosa
from audio_utilities import dynamic_range_compression
import numpy as np


# Find letter index from all_letters, e.g. "a" = 0
def character_to_index(character):
    return all_characters.find(character)


class LabelledMelDataset(Dataset):
    def __init__(self, root_dir, loader_fn):
        self.paths, self.labels = loader_fn(root_dir)
        self.root_dir = root_dir

    def label_to_list(self, label):
        return [all_characters.find(ch) for ch in label]

    def audiopath_to_mel(self, path):
        y, sr = librosa.load(path)

        # audio already normalised but just in case
        assert(np.min(y) >= -1)
        assert(np.max(y) <= 1)

        mel = librosa.feature.melspectrogram(y, sr,
                                             n_fft=1024,
                                             hop_length=256,
                                             n_mels=80,
                                             fmin=125,
                                             fmax=7600)

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
               torch.stack(padded_mels), torch.stack(padded_stop_tokens)


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
