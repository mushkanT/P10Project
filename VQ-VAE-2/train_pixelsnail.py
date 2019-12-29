import argparse
import numpy as np
import tensorflow as tf
try:
    from apex import amp

except ImportError:
    amp = None

import DataHandler
from PixelSNAIL import PixelSNAIL



def train(args, dataset, model, optimizer):
    losses = []
    for i, batch in enumerate(dataset):
        with tf.GradientTape() as tape:

            if args.hier == 'top':
                top = batch['top']
                target = top
                out, _ = model(top)

            elif args.hier == 'bottom':
                bottom = batch['bottom']
                target = bottom
                out, _ = model(bottom, condition=top)

            cross_entropy = tf.losses.CategoricalCrossentropy()
            loss = cross_entropy(target, out)
            losses.append(loss)
        trainable_varibles = model.trainable_variables
        grads = tape.gradient(loss, trainable_varibles)
        optimizer.apply_gradients(zip(grads, trainable_varibles))
    return losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=420)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_agument('--img_size', type=int, help='Image size as int')
    parser.add_argument('--path', type=str)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--run_folder', type=str)

    args = parser.parse_args()

    print(args)

    dataset = DataHandler.get_encodings(args.batch_size, shuffle=True, drop_remainder=True, path=args.path)


    top_input = 0
    bottom_input = 0
    if args.img_size == 32:
        top_input = 4
        bottom_input = 8
    elif args.img_size == 256:
        top_input = 32
        bottom_input = 64
    else:
        raise('Unsupported image size')

    if args.hier == 'top':
        model = PixelSNAIL(
            [top_input, top_input],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [bottom_input, bottom_input],
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


    optimizer = tf.keras.optimizers.Adam(lr=args.lr)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)


    for i in range(args.epoch):
        train(args, dataset, model, optimizer)
