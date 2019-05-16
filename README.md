# SRdarts: An implementation of DARTS on Super-Resolution Task

## Description

SRdarts is an implementation of DARTS on **Super-Resolution Task**, we use DARTS as the method baseline.



## Download

### Code

Clone the repository into your place and let's get started!

```bash
git clone https://github.com/GeniusGaryant/SRdarts.git
cd SRdarts
```

### Datasets

We use [DIV2K](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf) dataset as our train/valid dataset during the search.

For the test dataset, we use widely-used benchmark datasets as follows:

[Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)、[Set14](https://sites.google.com/site/romanzeyde/research-interests)、[B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)、[Urban100](https://sites.google.com/site/jbhuang0604/publications/struct_sr)



## Installation

### Requirements

- Python 3.6

- PyTorch 1.1 from a nightly release. Installation instructions can be found in [PyTorch](https://pytorch.org/get-started/locally/).
- torchvision
- cuda9.0

- numpy
- scikit-image
- imageio
- matplotlib
- tb-nightly
- future

### Step by step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do
conda create -n srdarts python=3.6
source activate srdarts

# install torchvision and pytorch1.1
pip install numpy torchvision_nightly
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

conda install scikit-image
conda install imageio

# install tb-nightly for tensorboard part
pip install tb-nightly
pip install future
```