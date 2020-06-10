# Improved Learning of Joint Distributions using Soft-CoupledGANs
## Tensorflow implementation of SCoGAN with FR, SL, CC and PL
### DISCLAIMER! - The code is still in a prototype stage and being cleaned for easier usage

_Abstract:_ <br>
In this project we analyse the joint distribution learning framework Coupled GAN and find that its imposed weight sharing constraint restricts the generators in learning the joint distribution over noisy and diverse datasets such as MNIST2SVHN, Apples2Oranges and Horses2Zebra. Through an experimental and research driven approach we propose to replace the strict weight sharing constraint with a softer coupling between generators in the shape of four regularisation terms. We call this type of model Soft-CoGAN (SCoGAN). These terms are (1) a feature regulariser which enforces generators to learn similar features, (2) a semantic loss based on classification of generated images such that the content of images are of the same class, (3) cycle consistency between latent vectors and (4) a perceptual loss which is a more advanced version of the feature regulariser using features from a pretrained deep classifier. Through experiments on different datasets we find that combinations of our proposed regularisers are able to provide a softer coupling that learns the joint distribution on MNIST2SVHN. However our approaches only achieve similar performance as CoGAN on the Apple2Oranges, Horse2Zebra and CelebA datasets. We discuss why and argue that through further tuning of hyperparameters our approaches could potentially surpass CoGAN performance.  

### Variants of the SCoGAN architecture
<p align="center">
<img alt="proposed MSG-GAN architecture" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/architecture.png"
width=90% />
</p>

<p>
The above figure describes the architecture of MSG-GAN for 
generating synchronized multi-scale images. Our method is 
based on the architecture proposed in proGAN, 
but instead of a progressively growing training scheme, 
includes connections from the intermediate
layers of the generator to the intermediate layers of the 
discriminator. The multi-scale images input to 
the discriminator are converted into spatial 
volumes which are concatenated with the corresponding 
activation volumes obtained from the main path of 
convolutional layers.
</p> <br>

<p>
For the discrimination process, appropriately downsampled 
versions of the real images are fed to corresponding layers 
of the discriminator as shown in the diagram (from above).
</p> <br>

<p align="center">
<img alt="synchronization explanation" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/synchronization.png"
     width=80% />
</p>
<br>

Above figure explains how, during training, all the layers 
in the MSG-GAN first synchronize colour-wise and subsequently 
improve the generated images at various scales. 
The brightness of the images across all layers (scales) 
synchronizes eventually

### Running the Code
Start the training by running the `main.py` script.
Refer to the following parameters for adapting to your own use:

    -h, --help            show this help message and exit
       --dataset DATASET     toy | mnist | cifar10 | lsun | frey | svhn
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
                             size of the latent vector
       --limit_dataset LIMIT_DATASET
                             limit dataset to one class
       --scale_data SCALE_DATA
                             Scale images in dataset to MxM
       --label_smooth LABEL_SMOOTH
                             Smooth the labels of the disc from 1 to 0 occasionally
       --input_noise INPUT_NOISE
                             Add gaussian noise to the discriminator inputs
       --purpose PURPOSE     purpose of this experiment
       --grayscale GRAYSCALE
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
       --feature_loss FEATURE_LOSS
                             Use vgg to extract features used for regularizing
       --fl_high_weight FL_HIGH_WEIGHT
                             Weight for high level feature similarity
       --fl_low_weight FL_LOW_WEIGHT
                             Weight for low level feature similarity
       --perceptual_loss PERCEPTUAL_LOSS
                             For using perceptual loss
       --style_weight STYLE_WEIGHT
                             If -1 use proportional to content weight, else use set
                             value
       --content_weight CONTENT_WEIGHT
                             Weight for content loss

##### Example of running a training
For training a SCoGAN network with semantic loss use the following arguments:

    $ python main.py --epochs=20000 \ 
                      --noise_dim=100 \
                      --dir=<path to wanted output directory> \
                      --cogan_data=mnist2svhn \
                      --classifier_path=<path to classifier used in semantic loss> \
                      --semantic_weight=10

We used 1 Tesla V100 GPUs of the 
DGX-2 machine for our experimentation.

### Generated samples on different datasets

<p align="center">
     <b> <b> :star: [NEW] :star: </b> CelebA HQ [1024 x 1024] (30K dataset)</b> <br>
     <img alt="CelebA-HQ" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/HQ_faces_sheet.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> <b> :star: [NEW] :star: </b> Oxford Flowers (improved samples) [256 x 256] (8K dataset)</b> <br>
     <img alt="oxford_big" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/flowers_sheet.png"
          width=80% />
     <img alt="oxford_variety" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/variety_flowers_sheet.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> CelebA HQ [256 x 256] (30K dataset)</b> <br>
     <img alt="CelebA-HQ" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/CelebA-HQ_sheet.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> LSUN Bedrooms [128 x 128] (3M dataset) </b> <br>
     <img alt="lsun_bedrooms" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/Bedrooms_sheet_new.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> CelebA [128 x 128] (200K dataset) </b> <br>
     <img alt="CelebA" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/faces_sheet.png"
          width=80% />
</p>
<br>

### Synchronized all-res generated samples
<p align="center">
     <b> Cifar-10 [32 x 32] (50K dataset)</b> <br>
     <img alt="cifar_allres" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/CIFAR10_allres_sheet.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> Oxford-102 Flowers [256 x 256] (8K dataset)</b> <br>
     <img alt="flowers_allres" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/FLowers_allres_sheet.png"
          width=80% />
</p>
<br>

## Authors

* **Patrick Alminde** - [Palminde](https://github.com/Palminde)
* **Markus Hald Juul-Nyholm** - [MarkusHald](https://github.com/MarkusHald)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Supervisor - Thomas Dyhre Nielsen, AAU.
