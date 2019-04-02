import params
from data import LabelledMelDataset, PadCollate, LJSpeechLoader, text_to_tensor, prepare_input, save_mels_to_png, load_from_files
from model import Tacotron2
import torch
import numpy as np
from torch import optim
from torch.utils import data
from torch.utils.data.dataset import random_split
import torch.nn as nn
import griffin_lim
from audio_utilities import dynamic_range_decompression
import sys


def train():
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    dataset_loader = LJSpeechLoader()
    dataset = LabelledMelDataset('resources/LJSpeech-1.1', dataset_loader)

    train_size = int((1.0 - params.validation_split) * dataset.__len__())
    val_size = int(params.validation_split * dataset.__len__())
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset, val_dataset, test_dataset = load_from_files('resources/LJSpeech-1.1/filelists', dataset)

    pad_collate = PadCollate()

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=params.batch_size,
                                   sampler=None,
                                   collate_fn=pad_collate,
                                   drop_last=True,
                                   num_workers=1)
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=params.batch_size,
                                 sampler=None,
                                 collate_fn=pad_collate,
                                 drop_last=True,
                                 num_workers=1)

    tacotron2 = Tacotron2()
    if params.use_gpu:
        tacotron2 = tacotron2.cuda()

    optimiser = optim.Adam(tacotron2.parameters(),
                           lr=params.learning_rate,
                           eps=params.epsilon,
                           weight_decay=params.weight_decay)

    criterion = nn.MSELoss()
    criterion_stop = nn.BCEWithLogitsLoss()

    start_epoch = 0

    if params.resume_from_checkpoint:
        checkpoint = torch.load(params.checkpoint_path)
        tacotron2.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    if params.inference:
        inference()

    tacotron2.train()

    for epoch in range(start_epoch, params.epochs):
        for i, batch in enumerate(train_loader):
            padded_texts, text_lengths, padded_mels, mel_lengths, padded_stop_tokens = prepare_input(batch)

            y_pred, y_pred_post, pred_stop_tokens = tacotron2(padded_texts, text_lengths, padded_mels)

            for b in range(params.batch_size):
                y_pred[b][mel_lengths[b]:] = 0
                y_pred_post[b][mel_lengths[b]:] = 0
                pred_stop_tokens[b][mel_lengths[b]:] = 1

            loss = criterion(y_pred, padded_mels) + criterion(y_pred_post, padded_mels) + \
                criterion_stop(pred_stop_tokens, padded_stop_tokens)
            loss.backward()

            print("Batch #" + str(i + 1))

            if (i + 1) % (params.effective_batch_size / params.batch_size) == 0:
                print("stepping backwards...")
                optimiser.step()
                optimiser.zero_grad()

        if epoch % params.checkpoint_skip == 0:
            # checkpoint the model
            if params.should_checkpoint:
                print("saving model...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': tacotron2.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict()},
                    params.checkpoint_path)

            print("generating audio...")
            mel_ref = griffin_lim.save_mel_to_wav(dynamic_range_decompression(padded_mels[-1][:, :mel_lengths[-1]].cpu().detach()),
                                            'Epoch ' + str(epoch + 1) + ' reference')
            mel_post = griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred_post[-1][:, :mel_lengths[-1]].cpu().detach()),
                                            'Epoch ' + str(epoch + 1) + ' post')

            # test if learning identity
            difference = padded_mels[-1][:, :mel_lengths[-1]] - y_pred_post[-1][:, :mel_lengths[-1]]
            mel_diff = griffin_lim.save_mel_to_wav(dynamic_range_decompression(difference.cpu().detach()), 'Difference')

            # save to png
            save_mels_to_png((mel_ref, mel_post, mel_diff), ("Reference", "Output", "Difference"), str(i))
                
            validate(tacotron2, val_loader, criterion, criterion_stop)


@torch.no_grad()  # no need for backpropagation -> speeds up computation
def validate(model, val_loader, criterion, criterion_stop):
    model.eval()

    avg_loss = 0.0
    for i, batch in enumerate(val_loader):
        padded_texts, text_lengths, padded_mels, mel_lengths, padded_stop_tokens = prepare_input(batch)

        y_pred, y_pred_post, pred_stop_tokens = model(padded_texts, text_lengths, padded_mels)
        loss = criterion(y_pred, padded_mels) + criterion(y_pred_post, padded_mels) + \
            criterion_stop(pred_stop_tokens, padded_stop_tokens)

        avg_loss += loss.item()

        print("Batch #" + str(i) + ": " + str(avg_loss))
    avg_loss = avg_loss / (i + 1)

    print("validation loss: " + str(avg_loss))

    model.train()


@torch.no_grad()  # no need for backpropagation -> speeds up computation
def inference(model):
    model.eval()

    text = text_to_tensor(params.inference_text)
    if params.use_gpu:
        text = text.cuda()

    y_pred, y_pred_post = model.inference(text)

    griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred[0].cpu().detach()), 'inference')
    griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred_post[0].cpu().detach()), 'inference post')

    sys.exit()


if __name__ == '__main__':
    train()  # TODO: currently no way of switching between training/inference modes
