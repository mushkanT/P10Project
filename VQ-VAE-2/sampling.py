import argparse
import tensorflow as tf
import vq_vae_model as vq
import PixelSNAIL
import numpy as np
import DataHandler

tf.random.set_seed(5)

def get_top_bottom_models(args, top_input, bottom_input):

    top_dummy = tf.zeros([1, 4, 4], dtype=tf.int64)
    bottom_dummy = tf.zeros([1, 8, 8], dtype=tf.int64)

    top_model = PixelSNAIL.PixelSNAIL(
        [top_input, top_input],
        512,
        256,
        5,
        4,
        4,
        256,
        dropout=0.1,
        n_out_res_block=0,
    )

    top_model(top_dummy)

    bottom_model = PixelSNAIL.PixelSNAIL(
        [bottom_input, bottom_input],
        512,
        256,
        5,
        4,
        4,
        256,
        attention=False,
        dropout=0.1,
        n_cond_res_block=3,
        cond_res_channel=256,
    )

    bottom_model(bottom_dummy, condition=top_dummy)

    return top_model, bottom_model


def sample_pixelsnail(model, batch, size, temp, condition=None):
    row = tf.Variable(tf.zeros([batch,size[0],size[1]], dtype=tf.int64))
    cache = {}

    for i in range(size[0]):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = tf.nn.softmax(out[:, i, j, :] / temp, 1)
            sample = tf.random.categorical(prob, 1)
            sample = tf.squeeze(sample, axis=-1)
            row[:, i, j].assign(sample)

    return row


def quantize(embeddings, indices):
    w = tf.keras.backend.transpose(embeddings.read_value())
    return tf.nn.embedding_lookup(w, indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=16, help='number of samples to generate')
    parser.add_argument('--img_size', type=int, help='size of images to sample')
    parser.add_argument('--channels', type=int, help='image channels')
    parser.add_argument('--vqvae_model', type=str, help='path to saved vqvae_model to decode image')
    parser.add_argument('--top_pixelCNN', type=str, help='path to saved weight of top pixelsnail model')
    parser.add_argument('--bottom_pixelCNN', type=str, help='path to saved weight of bottom pixelsnail model')
    parser.add_argument('--temp', type=float, default=1.0, help='unknown')
    parser.add_argument('output_path', type=str, help='where to save sample')

    args = parser.parse_args()

    vqvae_model = vq.VQVAEModel(args.img_size, args.channels)
    vqvae_model.load_model(args.vqvae_model)

    vq_top = vqvae_model.model.get_layer('vq_top')
    vq_bottom = vqvae_model.model.get_layer('vq_bottom')
    decoder = vqvae_model.model.get_layer('decoder')

    if args.img_size == 32:
        top_input = 4
        bottom_input = 8
    elif args.img_size == 256:
        top_input = 32
        bottom_input = 64
    else:
        raise ('Unsupported image size')



    top_model, bottom_model = get_top_bottom_models(args, top_input, bottom_input)

    top_model.load_weights(args.top_pixelCNN)
    bottom_model.load_weights(args.bottom_pixelCNN)

    top_sample = sample_pixelsnail(top_model, args.batch, [top_input, top_input], args.temp)
    bottom_sample = sample_pixelsnail(bottom_model, args.batch, [bottom_input, bottom_input], args.temp, condition=top_sample)

    #top_sample = tf.convert_to_tensor(np.load('/home/palminde/Desktop/torch_top.npy'))
    #bottom_sample = tf.convert_to_tensor(np.load('/home/palminde/Desktop/torch_bottom.npy'))

    #np.save('/home/palminde/Desktop/tf_top', top_sample.numpy())
    #np.save('/home/palminde/Desktop/tf_bottom', bottom_sample.numpy())

    decode_top = quantize(vq_top.embedding, top_sample)
    decode_bottom = quantize(vq_bottom.embedding, bottom_sample)

    decoded_sample = decoder([decode_top,decode_bottom])
    #decoded_sample = tf.clip_by_value(decoded_sample, -1, 1)

    np.save(args.output_path, decoded_sample.numpy())
