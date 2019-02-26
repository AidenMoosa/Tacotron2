import string

# training params
inference = False
inference_text = 'hello i am tacotron'
should_checkpoint = True
resume_from_checkpoint = True
checkpoint_path = 'out/model/model.p'
checkpoint_skip = 16
use_gpu = True
epochs = 10
batch_size = 8
effective_batch_size = 64
# optimiser parameters
learning_rate = 1e-3
epsilon = 1e-6
weight_decay = 1e-6
# model parameters
all_characters = string.printable
n_characters = len(all_characters)
embedding_dim = 512
# output parameters
audio_out_path = 'out/audio'
mel_out_path = 'out/mel'
# audio parameters
sampling_rate = 22050
n_mel_channels = 80
f_min = 125
f_max = 7600
n_fft = 1024
frame_hop = 256
window_function = 'hann'
griffin_lim_iters = 50
# wavenet parameters
n_dilation_conv_layers = 30
n_dilation_cycles = 3
n_upsampling_layers = 2
