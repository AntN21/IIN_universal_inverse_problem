# Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser

Visual images lie on a low-dimensional manifold, spanned by various natural deformations. Images on this manifold are approximately equally probable - at least locally. Probability of $x$ being a natural image, p(x), is zero everywhere except for x drawn from the manifold.
N(0, σ2) is drawn from an observation density, p(y), which is a Gaussian-blurred version of the image prior.
Moreover, the family of observation densities over different noise variances, pσ(y), forms a Gaussian scale-space representation of the prior analogous to the temporal evolution of a diffusion process

![alt text](test.png?raw=true)


## Pre-trained denoisers
The directory [denoisers](denoisers) contains denoisers trained for removing Gaussian noise from natural images with the objective of minimizing mean square error. The prior embedded in a denoiser depends on the architecture of the model as well as the data used during training. The [denoisers](denoisers)  directory contains a separate folder for each denoiser with a specific architecture. The code for each architecture can be found in [code/network.py](code/network.py). Under each architecure directory, there are multiple folders for the denoiser trained on a particular dataset, and a specific noise range. 

## Code
The code directory contains code for the [algorithm](code/algorithm_inv_prob.py), the pre-trained [denoisers architecture](code/network.py), and [helper functions](code/Utils_inverse_prob.py). 

## test_images
Multiple commonly used [color](test_images/color) and [grayscale](test_images/grayscale) image datasets are uploaded in the test_images directory.

## Requirements 
Here is the list of libraries you need to install to execute the code: 

python  3.7.6 \

numpy 1.19.4 \
skimage 0.17.2 \
matplotlib 1.19.4 \
PyTorch 1.7.0 \
argparse 1.1 \
os \
time \ 
sys \
gzip 
