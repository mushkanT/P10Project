import argparse
import pickle
import tqdm
import tensorflow as tf
import lmdb
from Data_Utils import CodeRow
import Utils.DataHandler as dh
from VQ_VAE_Model import VQVAEModel


def extract(lmdb_env, loader, model):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        for img, _, filename in loader:
            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for file, top, bottom in zip(filename, id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--path', type=str, default='')

    args = parser.parse_args()


    x_train, y_train = dh.mnist(norm_setting=1)
    dataset = tf.data.Dataset.from_tensor_slices((x_train))
    dataset = dataset.batch(128)
    loader = iter(dataset)
    print(next(loader))
    model = tf.saved_model.load(args.ckpt)
    model.summary()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, loader, model)
