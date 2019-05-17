# training parameters
use_wavenet = False
should_checkpoint = True
resume_from_checkpoint = True
checkpoint_path = 'out/model/model.p'
checkpoint_skip = 1
use_gpu = True
epochs = 400
batch_size = 8
effective_batch_size = 64
validation_split = 0.1
# inference parameterss
inference_text = 'The overwhelming majority of people in this country know how to sift the wheat from the chaff in ' \
                 'what they hear and what they read.'
decoder_step_limit = 1000
# teacher forcing parameters
teacher_forcing_mode = 'scheduled'
teacher_forcing_ratio = 1.
teacher_forcing_init_ratio = 1.
teacher_forcing_final_ratio = 0.
teacher_forcing_start_decay = 10000
teacher_forcing_decay_steps = 40000
teacher_forcing_decay_alpha = 0.
# optimiser parameters
learning_rate = 1e-3
learning_rate_min = 1e-5
learning_rate_decay_start = 256
gamma_decay = 0.9
epsilon = 1e-6
weight_decay = 1e-6
# model parameters
all_characters = [chr(i) for i in range(128)]
n_characters = len(all_characters)
embedding_dim = 512
# output parameters
audio_out_path = 'out/audio'
mel_out_path = 'out/mel'
# audio parameters
sampling_rate = 22050
n_mel_channels = 80
frame_hop = 275
f_min = 125
f_max = 7600
n_fft = 1024
griffin_lim_iters = 200
# wavenet parameters
n_dilation_conv_layers = 12
n_dilation_cycles = 2
n_upsampling_layers = 2
n_mol_components = 10
# misc parameters
seed = 42

