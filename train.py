import params
from data import LabelledMelDataset, PadCollate, LJSpeechLoader, text_to_tensor, prepare_input, save_mels_to_png, load_from_files
from model import Tacotron2
import random
import torch
import numpy as np
from torch import optim
from torch.utils import data
from torch.utils.data.dataset import random_split
import torch.nn as nn
import griffin_lim
from audio_utilities import dynamic_range_decompression
import sys
import os
import math


def calculate_teacher_forced_ratio(epoch):
    decay_steps = params.teacher_forcing_decay_steps
    alpha = params.teacher_forcing_decay_alpha
    init_tfr = params.teacher_forcing_init_ratio

    global_step = (epoch - 42.6) * 390

    # https://www.tensorflow.org/api_docs/python/tf/train/cosine_decay
    global_step = max(0, min(global_step, decay_steps))
    cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_teacher_forced_ratio = init_tfr * decayed

    return decayed_teacher_forced_ratio


def calculate_exponential_lr(epoch):
    # from pytorch docs
    #     def get_lr(self):
    #         return [base_lr * self.gamma ** self.last_epoch
    #                 for base_lr in self.base_lrs]

    offset_epoch = epoch - 70

    lr = max(params.learning_rate_min, params.learning_rate * params.gamma_decay ** offset_epoch)

    return lr


def train(use_multiple_gpus = False):
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    dataset_loader = LJSpeechLoader()
    dataset = LabelledMelDataset('resources/LJSpeech-1.1', dataset_loader)

    # train_size = int((1.0 - params.validation_split) * dataset.__len__())
    # val_size = int(params.validation_split * dataset.__len__())
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

    if use_multiple_gpus:
        tacotron2 = nn.DataParallel(tacotron2)

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
        inference(tacotron2)

    tacotron2.train()

    for epoch in range(start_epoch, params.epochs):
        teacher_forced_ratio = calculate_teacher_forced_ratio(epoch)
        lr = calculate_exponential_lr(epoch)

        print("Epoch #" + str(epoch) + ":")
        print("\tTraining model...")

        # TODO: start decay after 256 epochs
        # from stack overflow
        #for g in optimiser.param_groups:
            #g['lr'] = lr

        for i, batch in enumerate(train_loader):
            print("\tBatch #" + str(i + 1) + ": ")

            padded_texts, text_lengths, padded_mels, mel_lengths, padded_stop_tokens = prepare_input(batch)

            y_pred, y_pred_post, pred_stop_tokens = tacotron2(padded_texts, text_lengths, padded_mels, teacher_forced_ratio)

            for b in range(params.batch_size):
                y_pred[b][:][mel_lengths[b]:] = 0
                y_pred_post[b][:][mel_lengths[b]:] = 0
                pred_stop_tokens[b][mel_lengths[b]:] = 1

            loss = criterion(y_pred, padded_mels) + criterion(y_pred_post, padded_mels) + \
                criterion_stop(pred_stop_tokens, padded_stop_tokens)
            print("\t\tThe loss is... " + str(loss.item()))

            print("\t\tStepping backwards...")
            loss.backward()

            if (i + 1) % (params.effective_batch_size / params.batch_size) == 0:
                optimiser.step()
                optimiser.zero_grad()

        # write loss out
        with open('training_loss.txt', 'a+') as out:
            out.write(str(epoch) + '|' + str(loss.item()) + '\n')

        if epoch % params.checkpoint_skip == 0:
            # checkpoint the model
            if params.should_checkpoint:
                print("\tSaving model...")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': tacotron2.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict()},
                    params.checkpoint_path)
                
            validate(tacotron2, train_loader, val_loader, criterion, criterion_stop, epoch)


@torch.no_grad()  # no need for backpropagation -> speeds up computation
def validate(model, train_loader, val_loader, criterion, criterion_stop, epoch):
    model.eval()

    print("\tCalculating training loss...")
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
        print("\tBatch #" + str(i + 1) + ": ")

        padded_texts, text_lengths, padded_mels, mel_lengths, padded_stop_tokens = prepare_input(batch)

        y_pred, y_pred_post, pred_stop_tokens = model(padded_texts, text_lengths, padded_mels, teacher_forced_ratio=0.0)

        for b in range(params.batch_size):
            y_pred[b][:][mel_lengths[b]:] = 0
            y_pred_post[b][:][mel_lengths[b]:] = 0
            pred_stop_tokens[b][mel_lengths[b]:] = 1

        loss = criterion(y_pred, padded_mels) + criterion(y_pred_post, padded_mels) + \
            criterion_stop(pred_stop_tokens, padded_stop_tokens)
        train_loss += loss.item()

        print("\t\tLoss is... " + str(loss.item()))

    mel = griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred_post[0].cpu().detach()), 'Training ' + str(epoch))
    save_mels_to_png((mel, ), ("Mel", ), "Training " + str(epoch))

    train_loss = train_loss / (i + 1)

    # Write loss out
    with open('val_train_loss.txt', 'a+') as out:
        out.write(str(epoch) + "|" + str(train_loss) + '\n')

    print("\tCalculating test loss...")
    test_loss = 0.0
    for i, batch in enumerate(val_loader):
        print("\tBatch #" + str(i + 1) + ": ")

        padded_texts, text_lengths, padded_mels, mel_lengths, padded_stop_tokens = prepare_input(batch)

        y_pred, y_pred_post, pred_stop_tokens = model(padded_texts, text_lengths, padded_mels, teacher_forced_ratio=0.0)

        for b in range(params.batch_size):
            y_pred[b][:][mel_lengths[b]:] = 0
            y_pred_post[b][:][mel_lengths[b]:] = 0
            pred_stop_tokens[b][mel_lengths[b]:] = 1

        loss = criterion(y_pred, padded_mels) + criterion(y_pred_post, padded_mels) + \
            criterion_stop(pred_stop_tokens, padded_stop_tokens)
        test_loss += loss.item()

        print("\t\tLoss is... " + str(loss.item()))

    test_loss = test_loss / (i + 1)

    mel = griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred_post[0].cpu().detach()), 'Test ' + str(epoch))
    save_mels_to_png((mel, ), ("Mel", ), "Test " + str(epoch))

    print("\tSaving losses to file....")

    # Write loss out
    with open('val_test_loss.txt', 'a+') as out:
        out.write(str(epoch) + "|" + str(test_loss) + '\n')

    model.train()


@torch.no_grad()  # no need for backpropagation -> speeds up computation
def inference(model):
    model.eval()

    text = text_to_tensor(params.inference_text)
    if params.use_gpu:
        text = text.cuda()

    y_pred, y_pred_post = model.inference(text)

    mel = griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred[0].cpu().detach()), 'inference')
    mel_post = griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred_post[0].cpu().detach()), 'inference post')

    # save to png
    save_mels_to_png((mel, mel_post), ("Mel", "Post"), "Inference")

    sys.exit()


if __name__ == '__main__':
    '''
    use_multiple_gpus = False

    if len(sys.argv) > 1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    else:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            use_multiple_gpus = True
    '''

    # print(calculate_teacher_forced_ratio(112))

    train()  # TODO: currently no way of switching between training/inference modes
