import params
import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_utilities import MuLawEncoding, MuLawExpanding


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(params.embedding_dim, params.embedding_dim, kernel_size=5, padding=2, bias=True)
        torch.nn.init.xavier_uniform_(
            self.conv1.weight,
            gain=torch.nn.init.calculate_gain('relu'))
        self.conv1_bn = nn.BatchNorm1d(params.embedding_dim)

        self.conv2 = nn.Conv1d(params.embedding_dim, params.embedding_dim, kernel_size=5, padding=2, bias=True)
        torch.nn.init.xavier_uniform_(
            self.conv2.weight,
            gain=torch.nn.init.calculate_gain('relu'))
        self.conv2_bn = nn.BatchNorm1d(params.embedding_dim)

        self.conv3 = nn.Conv1d(params.embedding_dim, params.embedding_dim, kernel_size=5, padding=2, bias=True)
        torch.nn.init.xavier_uniform_(
            self.conv3.weight,
            gain=torch.nn.init.calculate_gain('relu'))
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

    def inference(self, embedded_input):
        inputs = F.dropout(F.relu(self.conv1_bn(self.conv1(embedded_input))), 0.5, self.training)
        inputs = F.dropout(F.relu(self.conv2_bn(self.conv2(inputs))), 0.5, self.training)
        inputs = F.dropout(F.relu(self.conv3_bn(self.conv3(inputs))), 0.5, self.training)
        inputs = inputs.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(inputs)

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
        torch.nn.init.xavier_uniform_(
            self.encoder_output_linear_proj.weight,
            gain=torch.nn.init.calculate_gain('tanh'))

        self.lstm_output_linear_proj = nn.Linear(1024, 128, bias=False)
        torch.nn.init.xavier_uniform_(
            self.lstm_output_linear_proj.weight,
            gain=torch.nn.init.calculate_gain('tanh'))

        self.location_conv = nn.Conv1d(1, 32, kernel_size=31, padding=15)
        torch.nn.init.xavier_uniform_(
            self.location_conv.weight,
            gain=torch.nn.init.calculate_gain('linear'))
        self.location_dense = nn.Linear(32, 128)
        torch.nn.init.xavier_uniform_(
            self.location_dense.weight,
            gain=torch.nn.init.calculate_gain('tanh'))

        self.e = nn.Linear(128, 1)
        torch.nn.init.xavier_uniform_(
            self.e.weight,
            gain=torch.nn.init.calculate_gain('linear'))

    def forward(self, encoder_output, processed_encoder_output, lstm_output, attention_weights_cum):
        processed_lstm_output = self.lstm_output_linear_proj(lstm_output)

        attention_weights_cum = attention_weights_cum.unsqueeze(1)
        processed_attention_weights = self.location_conv(attention_weights_cum)
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

        self.prenet_dense1 = nn.Linear(80, 256, bias=False)
        torch.nn.init.xavier_uniform_(
            self.prenet_dense1.weight,
            gain=torch.nn.init.calculate_gain('relu'))
        self.prenet_dense2 = nn.Linear(256, 256, bias=False)
        torch.nn.init.xavier_uniform_(
            self.prenet_dense2.weight,
            gain=torch.nn.init.calculate_gain('relu'))

        self.attention = Attention()

        self.lstm = nn.LSTM(256 + 512, 1024, num_layers=2)

        self.linear_proj = nn.Linear(1024 + 512, 80)
        torch.nn.init.xavier_uniform_(
            self.linear_proj.weight,
            gain=torch.nn.init.calculate_gain('linear'))

        self.stop_dense = nn.Linear(1024 + 512, 1)
        torch.nn.init.xavier_uniform_(
            self.stop_dense.weight,
            gain=torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, inputs, padded_mels):
        # can do this before
        processed_encoder_output = self.attention.encoder_output_linear_proj(inputs)

        #
        attention_weights_cum = padded_mels.new_zeros((params.batch_size, inputs.size(1)))
        attention_context = padded_mels.new_zeros((params.batch_size, params.embedding_dim))

        # prepend dummy mel frame
        padded_mels = torch.cat((padded_mels.new_zeros((params.batch_size, 80, 1)), padded_mels), dim=-1)

        # decoder loop
        predicted_mel, stop_tokens = [], []
        for decoder_step in range(padded_mels.size(-1) - 1):
            # grab previous mel frame as decoder input
            prev_mel = padded_mels[:, :, decoder_step]

            # convert mel frame into model-readable format
            prev_mel = F.dropout(F.relu(self.prenet_dense1(prev_mel)), p=0.5)
            prev_mel = F.dropout(F.relu(self.prenet_dense2(prev_mel)), p=0.5)

            #
            lstm_input = torch.cat((prev_mel, attention_context), dim=-1).unsqueeze(1)
            lstm_output, _ = self.lstm(lstm_input)

            #
            attention_context, attention_weights = self.attention(inputs,
                                                                  processed_encoder_output,
                                                                  lstm_output,
                                                                  attention_weights_cum)
            attention_weights_cum = attention_weights_cum + attention_weights

            #
            decoder_output = torch.cat((lstm_output, attention_context), dim=-1)
            attention_context = attention_context.squeeze(1)

            next_frame = self.linear_proj(decoder_output)
            next_frame = next_frame.squeeze(1)
            predicted_mel.append(next_frame)

            # stop token prediction
            stop_token = self.stop_dense(decoder_output)
            stop_token = stop_token.squeeze(-1)
            stop_token = stop_token.squeeze(-1)
            stop_tokens.append(stop_token)

        mel_output = torch.stack(predicted_mel)
        mel_output = mel_output.transpose(0, 1)

        stop_tokens_t = torch.stack(stop_tokens)
        stop_tokens_t = stop_tokens_t.transpose(0, 1)

        return mel_output, stop_tokens_t

    def inference(self, encoder_output):
        # can do this before
        processed_encoder_output = self.attention.encoder_output_linear_proj(encoder_output)

        #
        attention_weights_cum = encoder_output.new_zeros((encoder_output.size(0), encoder_output.size(1)))
        attention_context = encoder_output.new_zeros((encoder_output.size(0), params.embedding_dim))

        # prepend dummy mel frame
        prev_mel = encoder_output.new_zeros((encoder_output.size(0), 80))

        # decoder loop
        predicted_mel = [prev_mel]
        decoder_steps = 0
        while True:
            # convert mel frame into model-readable format
            prev_mel = predicted_mel[decoder_steps]
            prev_mel = F.dropout(F.relu(self.prenet_dense1(prev_mel)), p=0.5)
            prev_mel = F.dropout(F.relu(self.prenet_dense2(prev_mel)), p=0.5)

            #
            lstm_input = torch.cat((prev_mel, attention_context), dim=-1).unsqueeze(1)
            lstm_output, _ = self.lstm(lstm_input)

            #
            attention_context, attention_weights = self.attention(encoder_output,
                                                                  processed_encoder_output,
                                                                  lstm_output,
                                                                  attention_weights_cum)
            attention_weights_cum = attention_weights_cum + attention_weights

            #
            decoder_output = torch.cat((lstm_output, attention_context), dim=-1)
            attention_context = attention_context.squeeze(1)

            next_frame = self.linear_proj(decoder_output)
            next_frame = next_frame.squeeze(1)
            predicted_mel.append(next_frame)

            # stop token prediction
            stop_token = torch.sigmoid(self.stop_dense(decoder_output))
            decoder_steps = decoder_steps + 1
            if stop_token[0, 0, 0] > 0.5 or decoder_steps > 500:
                break

        mel_output = torch.stack(predicted_mel)
        mel_output = mel_output.transpose(0, 1)

        return mel_output


class Postnet(nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()

        self.postnet_conv1 = nn.Conv1d(80, 512, kernel_size=5, padding=2, bias=False)
        torch.nn.init.xavier_uniform_(
            self.postnet_conv1.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        self.postnet_conv1_bn = nn.BatchNorm1d(512)
        self.postnet_conv2 = nn.Conv1d(512, 512, kernel_size=5, padding=2, bias=False)
        torch.nn.init.xavier_uniform_(
            self.postnet_conv2.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        self.postnet_conv2_bn = nn.BatchNorm1d(512)
        self.postnet_conv3 = nn.Conv1d(512, 512, kernel_size=5, padding=2, bias=False)
        torch.nn.init.xavier_uniform_(
            self.postnet_conv3.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        self.postnet_conv3_bn = nn.BatchNorm1d(512)
        self.postnet_conv4 = nn.Conv1d(512, 512, kernel_size=5, padding=2, bias=False)
        torch.nn.init.xavier_uniform_(
            self.postnet_conv4.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        self.postnet_conv4_bn = nn.BatchNorm1d(512)
        self.postnet_conv5 = nn.Conv1d(512, 80, kernel_size=5, padding=2, bias=False)
        torch.nn.init.xavier_uniform_(
            self.postnet_conv5.weight,
            gain=torch.nn.init.calculate_gain('linear'))
        self.postnet_conv5_bn = nn.BatchNorm1d(80)

    def forward(self, inputs):
        inputs = F.dropout(torch.tanh(self.postnet_conv1_bn(self.postnet_conv1(inputs))), 0.5, self.training)
        inputs = F.dropout(torch.tanh(self.postnet_conv2_bn(self.postnet_conv2(inputs))), 0.5, self.training)
        inputs = F.dropout(torch.tanh(self.postnet_conv3_bn(self.postnet_conv3(inputs))), 0.5, self.training)
        inputs = F.dropout(torch.tanh(self.postnet_conv4_bn(self.postnet_conv4(inputs))), 0.5, self.training)
        inputs = F.dropout(self.postnet_conv5_bn(self.postnet_conv5(inputs)), 0.5, self.training)

        return inputs


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__( in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        result = super(CausalConv1d, self).forward(inputs)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class ResidualBlock(nn.Module):
    def __init__(self, dilation):
        super(ResidualBlock, self).__init__()

        self.filter_conv_layer = CausalConv1d(1, 1, 5, 1, dilation, 1, True)
        self.gate_conv_layer = CausalConv1d(1, 1, 5, 1, dilation, 1, True)

        self.residual_conv_layer = nn.Conv1d(1, 1)

    def forward(self, inputs):
        filter_inputs = self.filter_conv_layer(inputs)
        gate_inputs = self.gate_conv_layer(inputs)

        filter_activation = torch.tanh(filter_inputs)
        gate_activation = torch.sigmoid(gate_inputs)

        outputs = filter_activation * gate_activation
        residual_outputs = self.residual_conv_layer(outputs)
        residual_outputs = residual_outputs + inputs

        return outputs, residual_outputs


class Wavenet(nn.Module):
    def __init__(self):
        super(Wavenet, self).__init__()

        # We want to upsample the mel spectrogram to align it with the desired 16-bit 24KHz model --
        # Tacotron 2 paper specifies 2 upsampling layers
        self.t_conv_1 = nn.ConvTranspose1d(1, 1)
        self.t_conv_2 = nn.ConvTranspose1d(1, 1)

        self.causal_conv = CausalConv1d(1, 1)

        dilation_pow_limit = params.n_dilation_conv_layers / params.n_dilation_cycles
        self.residual_blocks = nn.ModuleList([ResidualBlock(pow(2, k) % dilation_pow_limit)
                                              for k in range(params.n_dilation_conv_layers)])

        skip_channels, end_channels, classes = 1

        self.conv_1 = nn.Conv1d(skip_channels, end_channels)
        self.conv_2 = nn.Conv1d(end_channels, classes)

    def forward(self, inputs):
        inputs = self.causal_conv(inputs)

        outputs = torch.zeros((1, 1))
        for residual_block in self.residual_blocks:
            skip_inputs, inputs = residual_block(inputs)
            outputs = outputs + skip_inputs

        outputs = self.conv_1(F.relu(outputs))
        outputs = self.conv_2(F.relu(outputs))
        mle = MuLawExpanding(quantization_channels=256)
        outputs = mle(outputs)
        outputs = F.softmax(outputs, dim=-1)

        return outputs


class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()

        self.embedding = nn.Embedding(params.n_characters, params.embedding_dim)
        # experimental
        torch.nn.init.xavier_normal_(self.embedding.weight)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, padded_texts, text_lengths, padded_mels):
        text_lengths = text_lengths.data

        embedded_inputs = self.embedding(padded_texts)
        embedded_inputs = embedded_inputs.transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        decoder_outputs, stop_tokens = self.decoder(encoder_outputs, padded_mels)
        decoder_outputs = decoder_outputs.transpose(1, 2)

        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = postnet_outputs + decoder_outputs

        print("\tIn Model: input size", padded_mels.size(),
              "output size", postnet_outputs.size())

        return decoder_outputs, postnet_outputs, stop_tokens

    def inference(self, text):
        text = text.unsqueeze(0)

        embedded_inputs = self.embedding(text)
        embedded_inputs = embedded_inputs.transpose(1, 2)

        encoder_outputs = self.encoder.inference(embedded_inputs)

        decoder_outputs = self.decoder.inference(encoder_outputs)
        decoder_outputs = decoder_outputs.transpose(1, 2)

        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = postnet_outputs + decoder_outputs

        return decoder_outputs, postnet_outputs
