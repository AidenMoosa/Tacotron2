import string

# training params
should_checkpoint = False
resume_from_checkpoint = False
checkpoint_path = 'out/model/model.p'
use_gpu = True
epochs = 10
batch_size = 8
effective_batch_size = 64
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
