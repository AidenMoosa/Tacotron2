import params
from data import LabelledMelDataset, PadCollate, LJSpeechLoader
from model import Tacotron2
import torch
import numpy as np
from torch import optim
from torch.utils import data
import torch.nn as nn
import griffin_lim


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

dataset_loader = LJSpeechLoader()
dataset = LabelledMelDataset('resources/LJSpeech-1.1', dataset_loader)

pad_collate = PadCollate()
data_loader = data.DataLoader(dataset,
                              batch_size=params.batch_size,
                              collate_fn=pad_collate,
                              drop_last=True)

tacotron2 = Tacotron2()
if params.use_gpu:
    tacotron2 = tacotron2.cuda()

optimiser = optim.Adam(tacotron2.parameters(),
                       lr=params.learning_rate,
                       weight_decay=params.weight_decay)

criterion = nn.MSELoss()

start_epoch = 0
loss = 0

if params.resume_from_checkpoint:
    checkpoint = torch.load(params.checkpoint_path)
    tacotron2.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']

tacotron2.train()

for epoch in range(start_epoch, params.epochs):
    for i, batch in enumerate(data_loader):
        tacotron2.zero_grad()

        padded_texts, text_lengths, padded_mels = batch

        if params.use_gpu:
            padded_texts = padded_texts.cuda().long()
            text_lengths = text_lengths.cuda().long()
            padded_mels = padded_mels.cuda().float()

        y_pred = tacotron2(padded_texts, text_lengths, padded_mels)
        loss = criterion(y_pred, padded_mels)

        print("batch #" + str(i + 1) + ": loss = " + str(loss.item()))

        loss.backward()
        optimiser.step()

        if i % 20 == 0:
            # checkpoint the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': tacotron2.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'loss': loss},
                params.checkpoint_path)

            griffin_lim.save_mel_to_wav(padded_mels[0], 'iter ' + str(i) + ' reference')
            griffin_lim.save_mel_to_wav(y_pred[0], 'iter ' + str(i))


