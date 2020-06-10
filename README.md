# Improved Learning of Joint Distributions using Soft-CoupledGANs
## Tensorflow implementation of SCoGAN with FR, SL, CC and PL

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
 Refer to the following parameters for tweaking for your own use:

    -h, --help            show this help message and exit
      --generator_file GENERATOR_FILE
                            pretrained weights file for generator
      --generator_optim_file GENERATOR_OPTIM_FILE
                            saved state for generator optimizer
      --shadow_generator_file SHADOW_GENERATOR_FILE
                            pretrained weights file for the shadow generator
      --discriminator_file DISCRIMINATOR_FILE
                            pretrained_weights file for discriminator
      --discriminator_optim_file DISCRIMINATOR_OPTIM_FILE
                            saved state for discriminator optimizer
      --images_dir IMAGES_DIR
                            path for the images directory
      --folder_distributed FOLDER_DISTRIBUTED
                            whether the images directory contains folders or not
      --flip_augment FLIP_AUGMENT
                            whether to randomly mirror the images during training
      --sample_dir SAMPLE_DIR
                            path for the generated samples directory
      --model_dir MODEL_DIR
                            path for saved models directory
      --loss_function LOSS_FUNCTION
                            loss function to be used: standard-gan, wgan-gp,
                            lsgan,lsgan-sigmoid,hinge, relativistic-hinge
      --depth DEPTH         Depth of the GAN
      --latent_size LATENT_SIZE
                            latent size for the generator
      --batch_size BATCH_SIZE
                            batch_size for training
      --start START         starting epoch number
      --num_epochs NUM_EPOCHS
                            number of epochs for training
      --feedback_factor FEEDBACK_FACTOR
                            number of logs to generate per epoch
      --num_samples NUM_SAMPLES
                            number of samples to generate for creating the grid
                            should be a square number preferably
      --checkpoint_factor CHECKPOINT_FACTOR
                            save model per n epochs
      --g_lr G_LR           learning rate for generator
      --d_lr D_LR           learning rate for discriminator
      --adam_beta1 ADAM_BETA1
                            value of beta_1 for adam optimizer
      --adam_beta2 ADAM_BETA2
                            value of beta_2 for adam optimizer
      --use_eql USE_EQL     Whether to use equalized learning rate or not
      --use_ema USE_EMA     Whether to use exponential moving averages or not
      --ema_decay EMA_DECAY
                            decay value for the ema
      --data_percentage DATA_PERCENTAGE
                            percentage of data to use
      --num_workers NUM_WORKERS
                            number of parallel workers for reading files

##### Example of running a training
For training a network at resolution `256 x 256`, 
use the following arguments:

    $ python train.py --depth=7 \ 
                      --latent_size=512 \
                      --images_dir=<path to images> \
                      --sample_dir=samples/exp_1 \
                      --model_dir=models/exp_1

Set the `batch_size`, `feedback_factor` and 
`checkpoint_factor` accordingly.
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

### Cite our work
    @article{karnewar2019msg,
      title={MSG-GAN: Multi-Scale Gradient GAN for Stable Image Synthesis},
      author={Karnewar, Animesh and Wang, Oliver and Iyengar, Raghu Sesha},
      journal={arXiv preprint arXiv:1903.06048},
      year={2019}
    }

### Other Contributors :smile:

<p align="center">
     <b> Cartoon Set [128 x 128] (10K dataset) by <a href="https://github.com/huangzh13">@huangzh13</a> </b> <br>
     <img alt="Cartoon_Set" src="https://github.com/huangzh13/BMSG-GAN/blob/dev/diagrams/cartoonset_sheet.png"
          width=80% />
</p>
<br>

### Thanks
Please feel free to open PRs here if 
you train on other datasets using this architecture. 
<br>

Best regards, <br>
@akanimax :)

## Authors

* **Patrick Alminde** - [Palminde](https://github.com/Palminde)
* **Markus Hald Juul-Nyholm** - [MarkusHald](https://github.com/MarkusHald)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Supervisor - Thomas Dyhre Nielsen, AAU.
