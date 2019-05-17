import params
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


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

        self.lstm = nn.LSTM(input_size=params.embedding_dim,
                            hidden_size=int(params.embedding_dim / 2),
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
        # (B, embedding_dim, MAX_LENGTH) -> (B, MAX_LENGTH, embedding_dim)
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

        self.location_conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=31, padding=15)
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

    def forward(self, encoder_output, text_lengths, processed_encoder_output, lstm_output, attention_weights_cum):
        # INPUT:
          # encoder_output: (B, MAX_LENGTH, embedding_dim)
          # processed encoder_output: (B, MAX_LENGTH, attention_dim)
          # lstm_output: (B, 1, 1024)
          # attention_weights_cum: (B, MAX_LENGTH)
        # OUTPUT:
          # attention_context: (B, 1, embedding_dim)
          # attention_weights: (B, MAX_LENGTH)

        # (B, 1, 1024) -> (B, 1, attention_dim)
        processed_lstm_output = self.lstm_output_linear_proj(lstm_output)

        # (B, (1), MAX_LENGTH) -> (B, 32, MAX_LENGTH)
        location_features = self.location_conv(attention_weights_cum.unsqueeze(1))
        location_features = location_features.transpose(1, 2)
        # (B, MAX_LENGTH, 32) -> (B, MAX_LENGTH, attention_dim)
        location_features = self.location_dense(location_features)

        energies = self.e(torch.tanh(processed_lstm_output + location_features + processed_encoder_output))
        energies = energies.squeeze(-1)

        for i in range(encoder_output.size(0)):
            energies[i, text_lengths[i]:] = float('-inf')

        attention_weights = F.softmax(energies, dim=-1)

        # (B, 1, MAX_LENGTH) * (B, MAX_LENGTH, embedding_dim) -> (B, 1, embedding_dim)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), encoder_output)

        return attention_context, attention_weights

    def inference(self, encoder_output, processed_encoder_output, lstm_output, attention_weights_cum):
        # INPUT:
          # encoder_output: (B, MAX_LENGTH, embedding_dim)
          # processed encoder_output: (B, MAX_LENGTH, attention_dim)
          # lstm_output: (B, 1, 1024)
          # attention_weights_cum: (B, MAX_LENGTH)
        # OUTPUT:
          # attention_context: (B, 1, embedding_dim)
          # attention_weights: (B, MAX_LENGTH)

        # (B, 1, 1024) -> (B, 1, attention_dim)
        processed_lstm_output = self.lstm_output_linear_proj(lstm_output)

        # (B, (1), MAX_LENGTH) -> (B, 32, MAX_LENGTH)
        location_features = self.location_conv(attention_weights_cum.unsqueeze(1))
        location_features = location_features.transpose(1, 2)
        # (B, MAX_LENGTH, 32) -> (B, MAX_LENGTH, attention_dim)
        location_features = self.location_dense(location_features)

        energies = self.e(torch.tanh(processed_lstm_output + location_features + processed_encoder_output))
        energies = energies.squeeze(-1)

        attention_weights = F.softmax(energies, dim=-1)
        # (B, 1, MAX_LENGTH) * (B, MAX_LENGTH, embedding_dim) -> (B, 1, embedding_dim)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), encoder_output)

        return attention_context, attention_weights


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.prenet_dense1 = nn.Linear(in_features=80, out_features=256, bias=False)
        torch.nn.init.xavier_uniform_(
            self.prenet_dense1.weight,
            gain=torch.nn.init.calculate_gain('relu'))
        self.prenet_dense2 = nn.Linear(in_features=256, out_features=256, bias=False)
        torch.nn.init.xavier_uniform_(
            self.prenet_dense2.weight,
            gain=torch.nn.init.calculate_gain('relu'))

        self.attention = Attention()

        self.lstm = nn.LSTM(input_size=256 + 512, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=False)

        self.linear_proj = nn.Linear(1024 + 512, 80)
        torch.nn.init.xavier_uniform_(
            self.linear_proj.weight,
            gain=torch.nn.init.calculate_gain('linear'))

        self.stop_dense = nn.Linear(1024 + 512, 1)
        torch.nn.init.xavier_uniform_(
            self.stop_dense.weight,
            gain=torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, inputs, text_lengths, padded_mels, teacher_forced_ratio=1.0):
        # project encoded input to latent space
        processed_encoder_output = self.attention.encoder_output_linear_proj(inputs)

        # zero initialise required variables
        attention_weights_cum = padded_mels.new_zeros((inputs.size(0), inputs.size(1)))
        lstm_output = padded_mels.new_zeros((inputs.size(0), 1, 1024))

        # prepend dummy mel frame
        padded_mels = torch.cat((padded_mels.new_zeros((inputs.size(0), 1, 80)), padded_mels), dim=1)

        # decoder loop
        predicted_mel = padded_mels.new_zeros((inputs.size(0), 80))
        predicted_mels, stop_tokens = [], []
        attention_weights_list = []
        for decoder_step in range(padded_mels.size(1) - 1):
            # grab previous mel frame as decoder input
            prev_mel = padded_mels[:, decoder_step, :]

            # experimental: using teacher-forced ratio to speed convergence
            if random.random() > teacher_forced_ratio:
                prev_mel = predicted_mel

            # convert mel frame into model-readable format
            prev_mel = F.dropout(F.relu(self.prenet_dense1(prev_mel)), p=0.5)
            prev_mel = F.dropout(F.relu(self.prenet_dense2(prev_mel)), p=0.5)

            #
            attention_context, attention_weights = self.attention(inputs,
                                                                  text_lengths,
                                                                  processed_encoder_output,
                                                                  lstm_output,
                                                                  attention_weights_cum)

            # evaluation code
            '''
            # for moving gaussian alignment
            ratio = padded_mels.size(1)/attention_weights.size(1)
            attention_weights = torch.zeros(attention_weights.size())

            # for random alignment
            # attention_weights = torch.rand(attention_weights.size())
            # attention_weights = F.softmax(attention_weights, dim=-1)

            # for either alignment
            attention_weights = attention_weights.cuda()

            # for moving gaussian alignment
            for b in range(params.batch_size):
                if decoder_step/ratio < text_lengths[b]:
                    if int(decoder_step/ratio) != 0:
                        attention_weights[b][int(decoder_step / ratio)-1] = 0.25

                    attention_weights[b][int(decoder_step/ratio)] = 0.5

                    if int(decoder_step/ratio) != attention_weights.size(1) - 1:
                        attention_weights[b][int(decoder_step / ratio)+1] = 0.25

            # for either alignment
            attention_context = torch.bmm(attention_weights.unsqueeze(1), inputs)
            '''

            attention_weights_cum = attention_weights_cum + attention_weights

            # evaluation code
            attention_weights_list.append(attention_weights)

            # (B, 1, 256 + 512)
            lstm_input = torch.cat((prev_mel.unsqueeze(1), attention_context), dim=-1)
            # (B, 1, 1024)
            lstm_output, _ = self.lstm(lstm_input)

            # (B, 1024 + embedding_dim)
            decoder_output = torch.cat((lstm_output, attention_context), dim=-1).squeeze(1)

            predicted_mel = self.linear_proj(decoder_output)
            predicted_mels.append(predicted_mel)

            # stop token prediction
            stop_token = self.stop_dense(decoder_output)
            stop_token = stop_token.squeeze(1)
            stop_tokens.append(stop_token)

        # evaluation code
        '''
        attention_weights_t = torch.stack(attention_weights_list)
        attention_weights_t = attention_weights_t.transpose(0, 1)
        attention_weights_t = attention_weights_t.transpose(1, 2)

        specshow(attention_weights_t.detach().cpu().numpy()[0], cmap=plt.cm.viridis)
        plt.show()
        '''

        mel_output = torch.stack(predicted_mels)
        mel_output = mel_output.transpose(0, 1)

        stop_tokens_t = torch.stack(stop_tokens)
        stop_tokens_t = stop_tokens_t.transpose(0, 1)

        return mel_output, stop_tokens_t

    def inference(self, encoder_output):
        # INPUT:
        #  encoder_output: (B, MAX_LENGTH, embedding_dim)

        # only slight changes to the training code

        processed_encoder_output = self.attention.encoder_output_linear_proj(encoder_output)

        attention_weights_cum = encoder_output.new_zeros((encoder_output.size(0), encoder_output.size(1)))
        lstm_output = encoder_output.new_zeros((encoder_output.size(0), 1, 1024))
        prev_mel = encoder_output.new_zeros((encoder_output.size(0), 80))

        predicted_mels = []
        decoder_step = 0
        while True:
            prev_mel = F.dropout(F.relu(self.prenet_dense1(prev_mel)), p=0.5)
            prev_mel = F.dropout(F.relu(self.prenet_dense2(prev_mel)), p=0.5)

            #
            attention_context, attention_weights = self.attention.inference(encoder_output,
                                                                            processed_encoder_output,
                                                                            lstm_output,
                                                                            attention_weights_cum)
            attention_weights_cum = attention_weights_cum + attention_weights

            #
            lstm_input = torch.cat((prev_mel.unsqueeze(1), attention_context), dim=-1)
            lstm_output, _ = self.lstm(lstm_input)

            decoder_output = torch.cat((lstm_output, attention_context), dim=-1).squeeze(1)

            predicted_mel = self.linear_proj(decoder_output)
            predicted_mels.append(predicted_mel)
            prev_mel = predicted_mel

            # stop token prediction
            stop_token = torch.sigmoid(self.stop_dense(decoder_output))
            decoder_step = decoder_step + 1

            if decoder_step > params.decoder_step_limit:  # should be stop_token[0, 0] > 0.5
                break

            print("Decoder step #" + str(decoder_step) + ": " + str(stop_token))

        mel_output = torch.stack(predicted_mels)
        mel_output = mel_output.transpose(0, 1)

        return mel_output


class Postnet(nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()

        # bias=False because it is encapsulated within the batch norm layer

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

        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        result = super(CausalConv1d, self).forward(inputs)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class ResidualBlock(nn.Module):
    def __init__(self, dilation):
        super(ResidualBlock, self).__init__()

        self.filter_conv_layer = CausalConv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=dilation, bias=True)
        self.gate_conv_layer = CausalConv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=dilation, bias=True)

        self.residual_conv_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.skip_conv_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)  # for skip parameters

    def forward(self, inputs):
        filter_inputs = self.filter_conv_layer(inputs)
        gate_inputs = self.gate_conv_layer(inputs)

        filter_activation = torch.tanh(filter_inputs)
        gate_activation = torch.sigmoid(gate_inputs)

        outputs = filter_activation * gate_activation

        residual_outputs = self.residual_conv_layer(outputs)
        residual_outputs = residual_outputs + inputs

        skip_outputs = self.skip_conv_layer(outputs)

        return residual_outputs, skip_outputs


class MixtureOfLogistics:
    def __init__(self, num_components=1):
        self.num_components = num_components

    # from https://pytorch.org/docs/stable/distributions.html#transformeddistribution
    def create_logistic_distribution(self, a, b):
        # Building a Logistic Distribution
        # X ~ Uniform(0, 1)
        # f = a + b * logit(X)
        # Y ~ f(X) ~ Logistic(a, b)
        base_distribution = D.Uniform(0, 1)
        transforms = [D.SigmoidTransform().inv, D.AffineTransform(loc=a, scale=b)]
        logistic = D.TransformedDistribution(base_distribution, transforms)
        return logistic

    def calculate_neg_log_probs(self, targets, weights):
        # INPUT:
        #   targets: B x 1
        #   weights: B x N_MOL_COMP x 3 (mix, loc, scale)

        # initialise distributions from weights
        distributions = [[self.create_logistic_distribution(weights[b][n][1], weights[b][n][2])
                          for n in range(weights.size(1))]
                         for b in range(weights.size(0))]

        # calculate log probs of mixture distribution given targets
        log_probs = torch.tensor([[distributions[b][n].log_prob(targets[b]) * weights[b][n][0]
                                  for n in range(weights.size(1))]
                                 for b in range(weights.size(0))])

        return -torch.sum(log_probs, dim=1)


# you may notice a lot of 1s, these are placeholders for when I can hopefully get some training done on WaveNet
class Wavenet(nn.Module):
    def __init__(self, skip_channels=1, end_channels=1, classes=1):
        super(Wavenet, self).__init__()

        # we need to create the mixture of logistic distributions to sample from (parameters are generated later)
        self.MoL = MixtureOfLogistics(params.n_mol_components)

        # we want to upsample the mel spectrogram to align it with the desired 16-bit 24KHz model ---
        # Tacotron 2 paper specifies 2 upsampling layers
        self.t_conv_1 = nn.ConvTranspose1d(in_channels=1, out_channels=1)
        self.t_conv_2 = nn.ConvTranspose1d(in_channels=1, out_channels=1)

        self.causal_conv = CausalConv1d(1, 1)

        dilation_pow_limit = params.n_dilation_conv_layers / params.n_dilation_cycles
        self.residual_blocks = nn.ModuleList([ResidualBlock(pow(2, k) % dilation_pow_limit)
                                              for k in range(params.n_dilation_conv_layers)])

        self.conv_1 = nn.Conv1d(skip_channels, end_channels)
        self.conv_2 = nn.Conv1d(end_channels, classes)

    def forward(self, inputs, targets):
        # input:
            # inputs: [B, n_mel_channels, n_mel_frames]
                # ground truth-aligned mel spectrograms
            # targets: [B, n_wav_frames]
                # ground truth wav files

        inputs = self.causal_conv(inputs)

        skip_outputs = inputs.new_zeros((1, 1))
        for residual_block in self.residual_blocks:
            r, s = residual_block(inputs)

            inputs = r
            skip_outputs = skip_outputs + s

        outputs = self.conv_1(F.relu(skip_outputs))
        outputs = self.conv_2(F.relu(outputs))

        # we're not using this (unmodified WaveNet)
        '''
        mle = MuLawExpanding(quantization_channels=256)
        outputs = mle(outputs)
        outputs = F.softmax(outputs, dim=-1)
        '''

        return outputs


class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()

        self.embedding = nn.Embedding(params.n_characters, params.embedding_dim)
        torch.nn.init.xavier_normal_(self.embedding.weight)

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, padded_texts, text_lengths, padded_mels, teacher_forced_ratio=1.0):
        embedded_inputs = self.embedding(padded_texts)
        embedded_inputs = embedded_inputs.transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        decoder_outputs, stop_tokens = self.decoder(encoder_outputs, text_lengths, padded_mels, teacher_forced_ratio)
        decoder_outputs = decoder_outputs.transpose(1, 2)

        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = postnet_outputs + decoder_outputs

        return decoder_outputs.transpose(1, 2), postnet_outputs.transpose(1, 2), stop_tokens

    def inference(self, text):
        text = text.unsqueeze(0)

        embedded_inputs = self.embedding(text)
        # (B, MAX_LENGTH, embedding_dim) -> (B, embedding_dim, MAX_LENGTH)
        embedded_inputs = embedded_inputs.transpose(1, 2)

        encoder_outputs = self.encoder.inference(embedded_inputs)

        decoder_outputs = self.decoder.inference(encoder_outputs)
        decoder_outputs = decoder_outputs.transpose(1, 2)

        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = postnet_outputs + decoder_outputs

        return decoder_outputs.transpose(1, 2), postnet_outputs.transpose(1, 2)
