import argparse

parser = argparse.ArgumentParser()

#Parser arguments
parser.add_argument('--dataset', type=str,            default = 'mnist',    help='mnist | cifar10 | Flickr1024 |')
parser.add_argument('--n_train', type=int,            default = 60000,      help='training set size, default to mnist')
parser.add_argument('--n_test', type=int,             default = 10000,      help='test set size, default to mnist')
parser.add_argument('--model', type=str,              default='VAE',        help='VAE | VQ_VAE_2 | GAN')
parser.add_argument('--noise_dim', type=int,          default = 100,        help='size of the latent vector')
parser.add_argument('--batch_size', type=int,         default = 100)
parser.add_argument('--epochs', type=int,             default = 1000)
parser.add_argument('--lr', type=float,               default = 1e-4,       help='learning rate')
parser.add_argument('--optim', type=str,              default='adam',       help='adam')
parser.add_argument('--num_samples_to_gen', type=int, default = 16)
parser.add_argument('--dir_out', type=str,            required=True,        help='output directory for images and models')
parser.add_argument('--mode', type=str,               default='build',      help='build | load')

args = parser.parse_args()

