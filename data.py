import os
from params import all_characters, n_characters
import torch
from torch.utils.data import Dataset
import librosa



# Find letter index from all_letters, e.g. "a" = 0
def character_to_index(character):
    return all_characters.find(character)

class LabelledMelDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.paths = []
        self.labels = []

        self.gather_labels()

    def gather_labels(self):
        paths = []
        labels = []
        for root, dirs, files in os.walk(self.root_dir, topdown=True):
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

        self.paths = paths
        self.labels = labels

    def label_to_onehot(self, label):
        onehots = [[0 for _ in range(n_characters)] for _ in range(len(label))]
        for i, ch in enumerate(label):
            onehots[i][all_characters.find(ch)] = 1
        return onehots

    def audiopath_to_mel(self, path):
        y, sr = librosa.load(path)
        mel = librosa.feature.melspectrogram(y, sr,
                                             n_fft=1024,
                                             hop_length=256,
                                             n_mels=80,
                                             fmin=125,
                                             fmax=7600)

        return mel

    def __getitem__(self, idx):
        text = self.label_to_onehot(self.labels[idx])
        mel = self.audiopath_to_mel(self.paths[idx])

        return text, mel

    def __len__(self):
        return len(self.paths)

# ideas taken from https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8
class PadCollate():
    def __init__(self):
        pass

    def pad_tensor(self, tensor, target_len, dim=-1):
        pad_shape = list(tensor.shape)
        pad_shape[dim] = target_len - tensor.size(dim)

        return torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype)], dim=dim)

    def __call__(self, batch):
        texts, mels = zip(*batch)

        max_text_len = max(map(lambda text: len(text), texts))
        padded_texts = [self.pad_tensor(torch.LongTensor(text), max_text_len, dim=0) for text in texts]

        max_mel_len = max(map(lambda mel: mel.shape[-1], mels))
        padded_mels = [self.pad_tensor(torch.from_numpy(mel), max_mel_len) for mel in mels]

        return torch.stack(padded_texts), torch.stack(padded_mels)

