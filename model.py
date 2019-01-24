import params
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(params.embedding_dim, params.embedding_dim, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm1d(params.embedding_dim)

        self.conv2 = nn.Conv1d(params.embedding_dim, params.embedding_dim, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm1d(params.embedding_dim)

        self.conv3 = nn.Conv1d(params.embedding_dim, params.embedding_dim, kernel_size=5, padding=2)
        self.conv3_bn = nn.BatchNorm1d(params.embedding_dim)

        self.lstm = nn.LSTM(params.embedding_dim, int(params.embedding_dim / 2), batch_first=True, bidirectional=True)

    def forward(self, embedded_inputs, input_lengths):
        inputs = F.dropout(F.relu(self.conv1_bn(self.conv1(embedded_inputs))), 0.5, self.training)
        inputs = F.dropout(F.relu(self.conv2_bn(self.conv2(inputs))), 0.5, self.training)
        inputs = F.dropout(F.relu(self.conv3_bn(self.conv3(inputs))), 0.5, self.training)

        # idea to pack came from NVIDIA
        inputs = nn.utils.rnn.pack_padded_sequence(inputs.transpose(1, 2), input_lengths, batch_first=True)
        outputs, _ = self.lstm(inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs


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

        self.memory_dense = nn.Linear(params.embedding_dim, 128, bias=False)

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
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.prenet_dense1 = nn.Linear(80, 256)
        self.prenet_dense2 = nn.Linear(256, 256)

        self.attention = Attention()

        self.lstm = nn.LSTM(1024 + 512, 1024, num_layers=2)

        self.linear_proj = nn.Linear(1024 + 512, 80)

        self.stop_dense = nn.Linear(1024 + 512, 1)

    def forward(self, inputs, padded_mels):
        # prepend dummy mel frame
        padded_mels = torch.cat((torch.zeros((5, 80, 1)), padded_mels), dim=2)
        print(padded_mels.size())

        # decoder loop
        for decoder_step in range(padded_mels.size(2) - 1):
            # grab previous mel frame as decoder input
            prev_mel = padded_mels[:, :, decoder_step]
            print(decoder_step)

            # convert mel frame into model-readable format
            prev_mel = F.dropout(F.relu(self.prenet_dense1(prev_mel)), p=0.5)
            prev_mel = F.dropout(F.relu(self.prenet_dense2(prev_mel)), p=0.5)

            #
            

        return inputs


class Postnet(nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()

        self.postnet_conv1 = nn.Conv1d(80, 512, kernel_size=5, padding=2)
        self.postnet_conv1_bn = nn.BatchNorm1d(512)
        self.postnet_conv2 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.postnet_conv2_bn = nn.BatchNorm1d(512)
        self.postnet_conv3 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.postnet_conv3_bn = nn.BatchNorm1d(512)
        self.postnet_conv4 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.postnet_conv4_bn = nn.BatchNorm1d(512)
        self.postnet_conv5 = nn.Conv1d(512, 80, kernel_size=5, padding=2)
        self.postnet_conv5_bn = nn.BatchNorm1d(80)

    def forward(self, inputs):
        inputs = F.dropout(torch.tanh(self.postnet_conv1_bn(self.postnet_conv1(inputs))), 0.5, self.training)
        inputs = F.dropout(torch.tanh(self.postnet_conv2_bn(self.postnet_conv2(inputs))), 0.5, self.training)
        inputs = F.dropout(torch.tanh(self.postnet_conv3_bn(self.postnet_conv3(inputs))), 0.5, self.training)
        inputs = F.dropout(torch.tanh(self.postnet_conv4_bn(self.postnet_conv4(inputs))), 0.5, self.training)
        inputs = F.dropout(self.postnet_conv5_bn(self.postnet_conv5(inputs)), 0.5, self.training)

        return inputs


class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()

        self.embedding = nn.Embedding(params.n_characters, params.embedding_dim)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, padded_texts, text_lengths, padded_mels):
        embedded_inputs = self.embedding(padded_texts).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        decoder_outputs = self.decoder(encoder_outputs, padded_mels)

        return decoder_outputs
