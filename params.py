# training params
inference = False
inference_text = 'Isobel likes to play in the sand'
should_checkpoint = True
resume_from_checkpoint = True
checkpoint_path = 'out/model/model.p'
checkpoint_skip = 1
use_gpu = True
epochs = 400
batch_size = 8
effective_batch_size = 64
validation_split = 0.1
seed = 42
#
teacher_forcing_mode = 'scheduled'  # Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
teacher_forcing_ratio = 1.  # Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
teacher_forcing_init_ratio = 1.  # initial teacher forcing ratio. Relevant if mode='scheduled'
teacher_forcing_final_ratio = 0.  # final teacher forcing ratio. (Set None to use alpha instead) Relevant if mode='scheduled'
teacher_forcing_start_decay = 10000  # starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
teacher_forcing_decay_steps = 40000  # Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
teacher_forcing_decay_alpha = 0.  # teacher forcing ratio decay rate. Defines the final tfr as a ratio of initial tfr. Relevant if mode='scheduled'
# optimiser parameters
learning_rate = 1e-3
learning_rate_min = 1e-5
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
n_mol_components = 10
