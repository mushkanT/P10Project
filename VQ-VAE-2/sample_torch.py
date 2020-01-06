import argparse
import os
import torch
from torchvision.utils import save_image
from vq_vae_model import VQVAEModel
from pixelsnail_torch import PixelSNAIL
import numpy as np
import tensorflow as tf


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in range(size[0]):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample
    return row


def load_model(model, checkpoint, device):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint), map_location=torch.device('cpu'))

    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = None

    elif model == 'pixelsnail_top':
        model = PixelSNAIL(
            [4, 4],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif model == 'pixelsnail_bottom':
        model = PixelSNAIL(
            [8, 8],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model


def quantize(embeddings, indices):
    w = tf.keras.backend.transpose(embeddings.read_value())
    return tf.nn.embedding_lookup(w, indices)


if __name__ == '__main__':
    device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--channels', type=int)
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('filename', type=str)

    args = parser.parse_args()

    vqvae_model = VQVAEModel(args.img_size, args.channels)
    vqvae_model.load_model(args.vqvae)

    vq_top = vqvae_model.model.get_layer('vq_top')
    vq_bottom = vqvae_model.model.get_layer('vq_bottom')
    decoder = vqvae_model.model.get_layer('decoder')

    model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)

    top_sample = sample_model(model_top, device, args.batch, [4, 4], args.temp)
    bottom_sample = sample_model(model_bottom, device, args.batch, [8, 8], args.temp, condition=top_sample)

    decode_top = quantize(vq_top.embedding, top_sample)
    decode_bottom = quantize(vq_bottom.embedding, bottom_sample)

    decoded_sample = decoder([decode_top, decode_bottom])

    decoded_sample = decoded_sample.clamp(-1, 1)
    save_image(decoded_sample, args.filename, normalize=True, range=(-1, 1))
