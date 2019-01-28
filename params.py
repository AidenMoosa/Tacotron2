import string

# training params
use_gpu = True
epochs = 10
batch_size = 2
# optimiser parameters
learning_rate = 1e-3
weight_decay = 1e-6
# model parameters
all_characters = string.printable
n_characters = len(all_characters)
embedding_dim = 512
# output parameters
audio_out_path = 'out/audio'
mel_out_path = 'out/mel'
