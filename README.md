# Improved Learning of Joint Distributions using Soft-CoupledGANs
## Tensorflow implementation of SCoGAN with FR, SL, CC and PL <br> DISCLAIMER! - The code is still in a prototype stage and is being cleaned for easier usage

_Abstract:_ <br>
In out master thesis we analyse the joint distribution learning framework Coupled GAN(CoGAN) and find that its imposed weight sharing constraint restricts the generators in learning the joint distribution over noisy and diverse datasets such as MNIST2SVHN, Apples2Oranges and Horses2Zebra. Through an experimental and research driven approach we propose to replace the strict weight sharing constraint with a softer coupling between generators in the shape of four regularisation terms. We call this type of model Soft-CoGAN (SCoGAN). These terms are (1) a feature regulariser which enforces generators to learn similar features, (2) a semantic loss based on classification of generated images such that the content of images are of the same class, (3) cycle consistency between latent vectors and (4) a perceptual loss which is a more advanced version of the feature regulariser using features from a pretrained deep classifier. Through experiments on different datasets we find that combinations of our proposed regularisers are able to provide a softer coupling that learns the joint distribution on MNIST2SVHN. However our approaches only achieve similar performance as CoGAN on the Apple2Oranges, Horse2Zebra and CelebA datasets. We discuss why and argue that through further tuning of hyperparameters our approaches could potentially surpass CoGAN performance.  


### Running the Code
Start the training by running the `main.py` script.
Refer to the following parameters for adapting to your own use:

    -h, --help            show this help message and exit
       --loss LOSS           wgan | ce
       --disc_penalty DISC_PENALTY
                             none | wgan-gp
       --gen_penalty GEN_PENALTY
                             weight | feature
       --batch_size BATCH_SIZE
       --epochs EPOCHS
       --disc_iters DISC_ITERS
       --clip CLIP           upper bound for clipping
       --penalty_weight_d PENALTY_WEIGHT_D
       --penalty_weight_g PENALTY_WEIGHT_G
       --lr_d LR_D
       --lr_g LR_G
       --b1 B1
       --b2 B2
       --optim_d OPTIM_D     adam | sgd | rms
       --optim_g OPTIM_G     adam | rms
       --num_samples_to_gen NUM_SAMPLES_TO_GEN
       --images_while_training IMAGES_WHILE_TRAINING
                             Every x epoch to print images while training
       --dir DIR             Directory to save images, models, weights etc
       --g_dim G_DIM         generator layer dimensions
       --d_dim D_DIM         discriminator layer dimensions
       --gan_type GAN_TYPE   64 | 128 | cifargan | cogan | classifier
       --noise_dim NOISE_DIM
                             size of the latent noise vector
       --input_noise INPUT_NOISE
                             Add gaussian noise to the discriminator inputs
       --purpose PURPOSE     Descriptive purpose of the execution
       --weight_decay WEIGHT_DECAY
       --bias_init BIAS_INIT
       --prelu_init PRELU_INIT
       --noise_type NOISE_TYPE
                             normal | uniform
       --weight_init WEIGHT_INIT
                             normal (0.02 mean)| xavier | he
       --g_arch G_ARCH       digit | rotate | 256 | face | digit_noshare |
                             face_noshare
       --d_arch D_ARCH       digit | rotate | 256 | face | digit_noshare |
                             face_noshare
       --cogan_data COGAN_DATA
                             mnist2edge | mnist2rotate | mnist2svhn |
                             mnist2negative | celeb_a | apple2orange | horse2zebra
                             | vangogh2photo
       --semantic_loss SEMANTIC_LOSS
                             Determines whether semantic loss is used
       --semantic_weight SEMANTIC_WEIGHT
                             Weight of the semantic loss term
       --classifier_path CLASSIFIER_PATH
                             Path to the classifier used for semantic loss
       --use_cycle USE_CYCLE
                             Turn on the cycle consistency loss
       --cycle_weight CYCLE_WEIGHT
                             Weight for the cycle gan loss
       --use_firstlayer USE_FIRSTLAYER
                             If using firstlayer corresponds with Torch, else with
                             caffe
       --shared_layers SHARED_LAYERS
                             Number of layers to calculate feature/weight
                             regularizer from
       --perceptual_loss PERCEPTUAL_LOSS
                             For using perceptual loss
       --style_weight STYLE_WEIGHT
                             If -1 use proportional to content weight, else use set
                             value
       --content_weight CONTENT_WEIGHT
                             Weight for content loss

##### Example of running a training
For training a SCoGAN network with semantic loss the following arguments can be used:

    $ python main.py --epochs=20000 \ 
                      --noise_dim=100 \
                      --dir=<path to wanted output directory> \
                      --cogan_data=mnist2svhn \
                      --classifier_path=<path to classifier used in semantic loss> \
                      --semantic_weight=10

We used 1 Tesla V100 GPUs of the 
DGX-2 machine for our experimentation.

### Additional generated samples on different datasets

<p align="center">
     <b> MNIST2SVHN_pruned [32x32] with SCoGAN-SL+FR</b> <br>
     <img alt="MNIST2SVHN" src="https://github.com/palminde/P9Project/blob/master/samples/Mnist2Svhn/sample1.png"
          width=80% />
</p>
<br>


## Authors

* **Patrick Alminde** - [Palminde](https://github.com/Palminde)
* **Markus Hald Juul-Nyholm** - [MarkusHald](https://github.com/MarkusHald)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Supervisor - Thomas Dyhre Nielsen, AAU.
