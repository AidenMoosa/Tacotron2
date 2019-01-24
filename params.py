import string

# training params
epochs = 10
batch_size = 5
# optimiser parameters
learning_rate = 1e-3
weight_decay = 1e-6
# model parameters
all_characters = string.printable
n_characters = len(all_characters)
embedding_dim = 512
