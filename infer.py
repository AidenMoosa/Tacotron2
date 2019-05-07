import torch
from model import Tacotron2
from data import text_to_tensor, save_mels_to_png, set_seed
import params
import griffin_lim
from audio_utilities import dynamic_range_decompression


@torch.no_grad()  # no need for backpropagation -> speeds up computation
def infer():
    model = Tacotron2()

    #
    checkpoint = torch.load(params.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    text = text_to_tensor(params.inference_text)

    if params.use_gpu:
        model = model.cuda()
        text = text.cuda()

    model.eval()  # batchnorm or dropout layers will work in eval model instead of training mode. \
                # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615

    y_pred, y_pred_post = model.inference(text)

    mel = griffin_lim.save_mel_to_wav(dynamic_range_decompression(y_pred_post[0].cpu().detach()), 'inference')
    save_mels_to_png((mel, ), ("Mel", ), "Inference", dpi=400)


if __name__ == '__main__':
    set_seed(params.seed)

    infer()
