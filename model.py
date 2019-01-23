import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import numpy as np

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

all_characters = string.printable
n_characters = len(all_characters)
embedding_dims = 512
batch_size = 5

character_embedding = nn.Embedding(n_characters, embedding_dims)

# Find letter index from all_letters, e.g. "a" = 0
def character_to_index(character):
    return all_characters.find(character)

def line_to_tensor(line):
    character_indices = torch.LongTensor([character_to_index(ch) for ch in line])
    return character_embedding(character_indices).t_()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

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

# want implement location-sensitive attention with scoring mechanism
# e_i,j = w^T tanh(W_s_i-1 + Vh_j + Uf_i,j + b)
# where w and b are vectors, W and V are matrices,
# s_i-1 is the (i - 1)-th state of the recurrent neural network to which we refer as the generator,
# h_j is the encoder output at j,
# j_i,j is the alignment information at j
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.query_dense = nn.Linear(1024, 128, bias=False)

        self.memory_dense = nn.Linear(embedding_dims, 128, bias=False)

        self.location_conv = nn.Conv1d(2, 32, kernel_size=31, padding=15)
        self.location_dense = nn.Linear(32, 128, bias=False)

        self.e = nn.Linear(128, 1, bias=False)


    def forward(self, attention_hidden_state, memory, processed_memory, attention_weights):
        processed_query = self.query_dense(attention_hidden_state)

        processed_attention_weights = self.location_conv(attention_weights)
        processed_attention_weights = processed_attention_weights.transpose(1, 2)
        processed_attention_weights = self.location_dense(processed_attention_weights)

        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))
        energies = energies.squeeze(-1)

        attention_weights = F.softmax(energies, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory) #same here
        attention_context = attention_context.squeeze(1) #same here

        return attention_context, attention_weights

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.prenet_dense1 = nn.Linear(80, 256)
        self.prenet_dense2 = nn.Linear(256, 256)

        self.attention = Attention()

        self.lstm = nn.LSTM(1024 + 512, 1024, num_layers=2)

        self.linear_proj = nn.Linear(1024 + 512, 80)

        self.postnet_conv1 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.postnet_conv1_bn = nn.BatchNorm1d(512)
        self.postnet_conv2 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.postnet_conv2_bn = nn.BatchNorm1d(512)
        self.postnet_conv3 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.postnet_conv3_bn = nn.BatchNorm1d(512)
        self.postnet_conv4 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.postnet_conv4_bn = nn.BatchNorm1d(512)
        self.postnet_conv5 = nn.Conv1d(512, 80, kernel_size=5, padding=2)
        self.postnet_conv5_bn = nn.BatchNorm1d(80)

        self.stop_dense = nn.Linear(1024 + 512, 1)

    def forward(self, input):
        input = F.dropout(F.relu(self.prenet_dense1(input)), p=0.5)
        input = F.dropout(F.relu(self.prenet_dense2(input)), p=0.5)

        input = F.dropout(torch.tanh(self.postnet_conv1_bn(self.postnet_conv1(input))), 0.5, self.training)
        input = F.dropout(torch.tanh(self.postnet_conv2_bn(self.postnet_conv2(input))), 0.5, self.training)
        input = F.dropout(torch.tanh(self.postnet_conv3_bn(self.postnet_conv3(input))), 0.5, self.training)
        input = F.dropout(torch.tanh(self.postnet_conv4_bn(self.postnet_conv4(input))), 0.5, self.training)
        input = F.dropout(self.postnet_conv5_bn(self.postnet_conv5(input)), 0.5, self.training)

        return input

class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        


    def forward(self, input):
        return input




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
