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

        self.lstm = nn.LSTM(params.embedding_dim, int(params.embedding_dim / 2),
                            batch_first=True,
                            bidirectional=True)

    def forward(self, embedded_inputs, input_lengths):
        inputs = F.dropout(F.relu(self.conv1_bn(self.conv1(embedded_inputs))), 0.5, self.training)
        inputs = F.dropout(F.relu(self.conv2_bn(self.conv2(inputs))), 0.5, self.training)
        inputs = F.dropout(F.relu(self.conv3_bn(self.conv3(inputs))), 0.5, self.training)
        inputs = inputs.transpose(1, 2)

        # idea to pack came from NVIDIA's implementation
        input_lengths = input_lengths.cpu().numpy()
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
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

        self.encoder_output_linear_proj = nn.Linear(params.embedding_dim, 128, bias=False)

        self.lstm_output_linear_proj = nn.Linear(1024, 128, bias=False)

        self.location_conv = nn.Conv1d(1, 32, kernel_size=31, padding=15, bias=False)
        self.location_dense = nn.Linear(32, 128, bias=False)

        self.e = nn.Linear(128, 1, bias=False)

    def forward(self, encoder_output, processed_encoder_output, lstm_output, attention_weights_cum):
        processed_lstm_output = self.lstm_output_linear_proj(lstm_output)

        attention_weights_cum = attention_weights_cum.unsqueeze(1)
        processed_attention_weights = self.location_conv(attention_weights_cum.cuda())
        processed_attention_weights = processed_attention_weights.transpose(1, 2)
        processed_attention_weights = self.location_dense(processed_attention_weights)

        energies = self.e(torch.tanh(processed_lstm_output + processed_attention_weights + processed_encoder_output))
        energies = energies.squeeze(-1)

        attention_weights = F.softmax(energies, dim=1)
        attention_weights = attention_weights.unsqueeze(1)
        attention_context = torch.bmm(attention_weights, encoder_output)
        attention_weights = attention_weights.squeeze(1)

        return attention_context, attention_weights


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.prenet_dense1 = nn.Linear(80, 256)
        self.prenet_dense2 = nn.Linear(256, 256)

        self.attention = Attention()

        self.lstm = nn.LSTM(256 + 512, 1024, num_layers=2)

        self.linear_proj = nn.Linear(1024 + 512, 80)

        self.stop_dense = nn.Linear(1024 + 512, 1)

    def forward(self, inputs, padded_mels):
        # can do this before
        processed_encoder_output = self.attention.encoder_output_linear_proj(inputs)

        #
        attention_weights_cum = torch.zeros((params.batch_size, inputs.size(1)))
        attention_context = torch.zeros((params.batch_size, params.embedding_dim))

        # prepend dummy mel frame
        padded_mels = torch.cat((torch.zeros((params.batch_size, 80, 1)).cuda(), padded_mels), dim=-1)

        # decoder loop
        predicted_mel = []
        for decoder_step in range(padded_mels.size(2) - 1):
            # grab previous mel frame as decoder input
            prev_mel = padded_mels[:, :, decoder_step]

            # convert mel frame into model-readable format
            prev_mel = F.dropout(F.relu(self.prenet_dense1(prev_mel)), p=0.5)
            prev_mel = F.dropout(F.relu(self.prenet_dense2(prev_mel)), p=0.5)

            #
            lstm_input = torch.cat((prev_mel, attention_context.cuda()), dim=-1).unsqueeze(1)
            lstm_output, _ = self.lstm(lstm_input)

            #
            attention_context, attention_weights = self.attention(inputs,
                                                                  processed_encoder_output,
                                                                  lstm_output,
                                                                  attention_weights_cum)
            attention_weights_cum = attention_weights_cum.cuda() + attention_weights.cuda()

            # maybe wrong
            decoder_output = torch.cat((lstm_output, attention_context), dim=-1)
            attention_context = attention_context.squeeze(1)

            next_frame = self.linear_proj(decoder_output)
            next_frame = next_frame.squeeze(1)
            predicted_mel.append(next_frame)

        mel_output = torch.stack(predicted_mel)
        mel_output = mel_output.transpose(0, 1)
        return mel_output


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
        text_lengths = text_lengths.data

        embedded_inputs = self.embedding(padded_texts).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        decoder_outputs = self.decoder(encoder_outputs, padded_mels)

        postnet_outputs = self.postnet(decoder_outputs.transpose(1, 2))
        postnet_outputs = postnet_outputs.transpose(1, 2)
        postnet_outputs = postnet_outputs + decoder_outputs
        postnet_outputs = postnet_outputs.transpose(1, 2)

        return postnet_outputs
