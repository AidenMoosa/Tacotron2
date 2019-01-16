import torch
import torch.nn as nn
import torch.nn.functional as F
import string
from data import AnnotatedVoiceDataset
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# much of this code adapted from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

all_characters = string.printable
n_characters = len(all_characters)
embedding_dims = 512
batch_size = 5

#character_embedding = torch.randn([n_characters, n_dims], requires_grad=True)
character_embedding = nn.Embedding(n_characters, embedding_dims)

# Find letter index from all_letters, e.g. "a" = 0
def character_to_index(character):
    return all_characters.find(character)


'''
# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def character_to_tensor(character):
    tensor = torch.zeros(1, n_characters
                         )
    tensor[0][character_to_index(character)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_characters)
    for li, character in enumerate(line):
        tensor[li][0][character_to_index(character)] = 1
    return tensor
'''

'''
def line_to_tensor(line):
    tensor = torch.zeros(len(line), embedding_dims)
    for ci, character in enumerate(line):
        tensor[ci] = character_embedding(character_to_index(character))
    return tensor.transpose(0, 1)
'''

def line_to_tensor(line):
    character_indices = torch.LongTensor([character_to_index(ch) for ch in line])
    return character_embedding(character_indices).t_()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv1d(embedding_dims, embedding_dims, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm1d(embedding_dims)

        self.conv2 = nn.Conv1d(embedding_dims, embedding_dims, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm1d(embedding_dims)

        self.conv3 = nn.Conv1d(embedding_dims, embedding_dims, kernel_size=5, padding=2)
        self.conv3_bn = nn.BatchNorm1d(embedding_dims)

        self.lstm = nn.LSTM(embedding_dims, int(embedding_dims / 2), batch_first=True, bidirectional=True)



    def forward(self, input):
        output = F.dropout(F.relu(self.conv1_bn(self.conv1(input))), 0.5, self.training)
        output = F.dropout(F.relu(self.conv2_bn(self.conv2(output))), 0.5, self.training)
        output = F.dropout(F.relu(self.conv3_bn(self.conv3(output))), 0.5, self.training)
        output = self.lstm(output.transpose(1, 2))

        return output


avd = AnnotatedVoiceDataset('resources')

model = Model()

for i in range(avd.__len__()):
    name, label = avd.__getitem__(i)

    input_tensor = line_to_tensor(label).unsqueeze(0)

    output, _ = model(input_tensor)
    print(output.size())

'''
data, sampling_rate = librosa.load(name)

S = librosa.feature.melspectrogram(y=data, sr=sampling_rate)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S,
                                             ref=np.max),
                                             y_axis='mel', fmax=8000,
                                             x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
librosa.display.waveplot(data, sr=sampling_rate)
plt.show()
'''
