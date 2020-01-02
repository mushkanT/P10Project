import argparse
import tensorflow as tf
from vq_vae_model import VQVAEModel
import DataHandler
import numpy as np
import pickle


def extract(dataset, model, output_path):
    encodings = []
    for i,batch in enumerate(dataset):
        print('Getting encodings from batch:' + str(i))
        out = model.model(batch)
        encodings.append({'top':out[1]['encoding_indices'].numpy(), 'bottom':out[2]['encoding_indices'].numpy()})
    print('saving encodings')
    np.save(output_path + 'newencodings', encodings)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--channels', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--data_path', type=str, default='mnist')

    args = parser.parse_args()
    print(args)

    dataset = DataHandler.get_dataset(args.batch_size, data_name=args.data_path, pad_to_32=True, shuffle=False, drop_remainder=False)

    model = VQVAEModel(args.img_size, args.channels)
    model.load_model(args.model_path)

    extract(dataset, model, args.out_path)
