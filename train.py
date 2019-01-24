import params
from data import LabelledMelDataset, PadCollate
from model import Tacotron2
import torch
import numpy as np
from torch import optim
from torch.utils import data
import torch.nn as nn

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

dataset = LabelledMelDataset('resources')
pad_collate = PadCollate()

data_loader = data.DataLoader(dataset,
                              batch_size=5,
                              collate_fn=pad_collate,
                              drop_last=True)

tacotron2 = Tacotron2()
tacotron2.train()

optimiser = optim.Adam(tacotron2.parameters(),
                       lr=params.learning_rate,
                       weight_decay=params.weight_decay)

criterion = nn.MSELoss()

for _ in range(params.epochs):
    for batch in data_loader:
        tacotron2.zero_grad()

        padded_texts, text_lengths, padded_mels = batch
        y_pred = tacotron2(padded_texts, text_lengths, padded_mels)
        loss = criterion(y_pred, padded_mels)

        print("Loss: " + str(loss.item()))

        loss.backward()

        optimiser.step()
