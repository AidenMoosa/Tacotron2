import params
from data import LabelledMelDataset, PadCollate, LJSpeechLoader, prepare_input, load_from_files, set_seed, save_mels_to_png
from griffin_lim import save_mel_to_wav
from model import Tacotron2
import torch
from torch import optim
from torch.utils import data
import torch.nn as nn
import math
from audio_utilities import dynamic_range_decompression
import griffin_lim


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

    offset_epoch = epoch - 260

    lr = max(params.learning_rate_min, params.learning_rate * params.gamma_decay ** offset_epoch)

    return lr


def calculate_loss(model, batch, teacher_forced_ratio, criterion, criterion_stop):
    padded_texts, text_lengths, padded_mels, mel_lengths, padded_stop_tokens = prepare_input(batch)

    y_pred, y_pred_post, pred_stop_tokens = model(padded_texts, text_lengths, padded_mels, teacher_forced_ratio)

    for b in range(params.batch_size):
        y_pred[b][:][mel_lengths[b]:] = 0
        y_pred_post[b][:][mel_lengths[b]:] = 0
        pred_stop_tokens[b][mel_lengths[b]:] = 1

    loss = criterion(y_pred, padded_mels) + criterion(y_pred_post, padded_mels) + \
        criterion_stop(pred_stop_tokens, padded_stop_tokens)

    mel = griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred_post[0].cpu().detach()), 'BIG OLD TEST')
    save_mels_to_png((mel, ), ("Title", ), "zzzz")
    save_mel_to_wav(mel, "zzz")

    return loss


def train(use_multiple_gpus=False):
    model = Tacotron2()

    dataset_loader = LJSpeechLoader()
    dataset = LabelledMelDataset('resources/LJSpeech-1.1', dataset_loader)

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

    '''
    if use_multiple_gpus:
        model = nn.DataParallel(model)
    '''

    if params.use_gpu:
        model = model.cuda()

    optimiser = optim.Adam(model.parameters(),
                           lr=params.learning_rate,
                           eps=params.epsilon,
                           weight_decay=params.weight_decay)

    criterion = nn.MSELoss()
    criterion_stop = nn.BCEWithLogitsLoss()

    start_epoch = 0
    if params.resume_from_checkpoint:
        checkpoint = torch.load(params.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    model.train()
    for epoch in range(start_epoch, params.epochs):
        teacher_forced_ratio = calculate_teacher_forced_ratio(epoch)
        teacher_forced_ratio = 0.25
        lr = calculate_exponential_lr(epoch)

        print("Epoch #" + str(epoch) + ":")
        print("\tTraining model...")

        # TODO: start decay after 256 epochs
        # from stack overflow
        #for g in optimiser.param_groups:
            #g['lr'] = lr

        for i, batch in enumerate(train_loader):
            print("\tBatch #" + str(i + 1) + ": ")

            loss = calculate_loss(model, batch, teacher_forced_ratio, criterion, criterion_stop)
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
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict()},
                    params.checkpoint_path)

            validate(model, train_loader, val_loader, criterion, criterion_stop, epoch)


@torch.no_grad()  # no need for backpropagation -> speeds up computation
def validate(model, train_loader, val_loader, criterion, criterion_stop, epoch):
    model.eval()

    print("\tCalculating training loss...")
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
        print("\tBatch #" + str(i + 1) + ": ")

        loss = calculate_loss(model, batch, 0.0, criterion, criterion_stop)
        train_loss += loss.item()

        print("\t\tLoss is... " + str(loss.item()))
    train_loss = train_loss / (i + 1)

    with open('val_train_loss.txt', 'a+') as out:
        out.write(str(epoch) + "|" + str(train_loss) + '\n')

    print("\tCalculating test loss...")
    test_loss = 0.0
    for i, batch in enumerate(val_loader):
        print("\tBatch #" + str(i + 1) + ": ")

        loss = calculate_loss(model, batch, 0.0, criterion, criterion_stop)
        test_loss += loss.item()

        print("\t\tLoss is... " + str(loss.item()))
    test_loss = test_loss / (i + 1)

    with open('val_test_loss.txt', 'a+') as out:
        out.write(str(epoch) + "|" + str(test_loss) + '\n')

    model.train()


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

    set_seed(params.seed)

    print(calculate_teacher_forced_ratio(145))

    train()
