import params
from data import LabelledMelDataset, PadCollate, LJSpeechLoader
from model import Tacotron2
import torch
import numpy as np
from torch import optim
from torch.utils import data
import torch.nn as nn
import griffin_lim
from audio_utilities import dynamic_range_decompression

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
    criterion_stop = nn.MSELoss()

    start_epoch = 0
    start_i = 0
    loss = 0

    if params.resume_from_checkpoint:
        checkpoint = torch.load(params.checkpoint_path)
        tacotron2.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_i = checkpoint['iteration']

    tacotron2.train()

    for epoch in range(start_epoch, params.epochs):
        for i, batch in enumerate(data_loader):
            padded_texts, text_lengths, padded_mels, padded_stop_tokens = batch

            if params.use_gpu:
                padded_texts = padded_texts.cuda().long()
                text_lengths = text_lengths.cuda().long()
                padded_mels = padded_mels.cuda().float()
                padded_stop_tokens = padded_stop_tokens.cuda().float()

            y_pred, y_pred_post, pred_stop_tokens = tacotron2(padded_texts, text_lengths, padded_mels)
            loss = criterion(y_pred, padded_mels) + criterion(y_pred_post, padded_mels) + \
                   criterion_stop(padded_stop_tokens, pred_stop_tokens)
            loss.backward()

            print("batch #" + str(start_i + i + 1) + ": loss = " + str(loss.item()))

            if (start_i + i + 1) % (params.effective_batch_size / params.batch_size) == 0:
                print("stepping backwards...")
                optimiser.step()
                optimiser.zero_grad()

            if (start_i + i + 1) % params.checkpoint_skip == 0:
                # checkpoint the model
                print("saving model...")
                if params.should_checkpoint:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': tacotron2.state_dict(),
                        'optimizer_state_dict': optimiser.state_dict(),
                        'iteraion': start_i + i},
                        params.checkpoint_path)

                griffin_lim.save_mel_to_wav(dynamic_range_decompression(padded_mels[0].cpu().detach()), 'iter ' + str(start_i + i) + ' reference')
                griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred[0].cpu().detach()), 'iter ' + str(start_i + i))
                griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred_post[0].cpu().detach()), 'iter ' + str(start_i + i) + 'post')

