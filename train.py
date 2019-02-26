import params
from data import LabelledMelDataset, PadCollate, LJSpeechLoader, text_to_tensor
from model import Tacotron2
import torch
import numpy as np
from torch import optim
from torch.utils import data
import torch.nn as nn
import griffin_lim
from audio_utilities import dynamic_range_decompression
import sys

if __name__ == '__main__':
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
    start_i = 0
    loss = 0

    if params.resume_from_checkpoint:
        checkpoint = torch.load(params.checkpoint_path)
        tacotron2.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_i = checkpoint['iteration']

    if not params.inference:
        tacotron2.train()
    else:
        tacotron2.eval()

    if params.inference:
        text = text_to_tensor(params.inference_text)

        if params.use_gpu:
            text = text.cuda().long()

        text.requires_grad = False

        y_pred, y_pred_post = tacotron2.inference(text)

        griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred[0].cpu().detach()), 'inference')
        griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred_post[0].cpu().detach()), 'inference post')

        sys.exit()

    for epoch in range(start_epoch, params.epochs):
        for i, batch in enumerate(data_loader):
            padded_texts, text_lengths, padded_mels, mel_lengths, padded_stop_tokens = batch

            if params.use_gpu:
                padded_texts = padded_texts.cuda().long()
                text_lengths = text_lengths.cuda().long()
                padded_mels = padded_mels.cuda().float()
                mel_lengths = mel_lengths.cuda().long()
                padded_stop_tokens = padded_stop_tokens.cuda().float()

            padded_texts.requires_grad = False
            text_lengths.requires_grad = False
            padded_mels.requires_grad = False
            mel_lengths.requires_grad = False
            padded_stop_tokens.requires_grad = False

            y_pred, y_pred_post, pred_stop_tokens = tacotron2(padded_texts, text_lengths, padded_mels)

            for b in range(params.batch_size):
                y_pred[b][mel_lengths[b]:] = 0
                y_pred_post[b][mel_lengths[b]:] = 0
                pred_stop_tokens[b][mel_lengths[b]:] = 1

            loss = criterion(y_pred, padded_mels) + criterion(y_pred_post, padded_mels) + \
                criterion_stop(pred_stop_tokens, padded_stop_tokens)
            loss.backward()

            print("Batch #" + str(start_i + i + 1) + ": loss = " + str(loss.item()))

            if (start_i + i + 1) % (params.effective_batch_size / params.batch_size) == 0:
                print("stepping backwards...")
                optimiser.step()
                optimiser.zero_grad()

            if (start_i + i + 1) % params.checkpoint_skip == 0:
                # checkpoint the model
                if params.should_checkpoint:
                    print("saving model...")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': tacotron2.state_dict(),
                        'optimizer_state_dict': optimiser.state_dict(),
                        'iteration': start_i + i + 1},
                        params.checkpoint_path)

                print("generating audio...")
                griffin_lim.save_mel_to_wav(dynamic_range_decompression(padded_mels[-1][:, :mel_lengths[-1]].cpu().detach()),
                                            'Iteration ' + str(start_i + i + 1) + ' reference')
                griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred_post[-1][:, :mel_lengths[-1]].cpu().detach()),
                                            'Iteration ' + str(start_i + i + 1) + ' post')

                # test if learning identity
                difference = padded_mels[-1][:, :mel_lengths[-1]] - y_pred_post[-1][:, :mel_lengths[-1]]
                griffin_lim.save_mel_to_wav(dynamic_range_decompression(difference.cpu().detach()), 'Difference')

