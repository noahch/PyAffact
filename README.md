# PyAffact
PyTorch implementation of the Affact Paper

Abstract

## Table of contents

* [Table of contents](#table-of-contents)
* [Quick start](#quick-start)
  + [*Configuration file*](#configuration-file)
  + [*WandB*](#wandb)
* [Pretrained models](#pretrained-models)
* [Training a model](#training-a-model)
  + [*Resnet51-S*](#training-resnet51-s)
  + [*AFFACT*](#training-affact)
  + [*Resnet152*](#training-resnet152)
* [Evaluation of a model](#evaluation-of-a-model)
  + [*Generate test datasets*](#generate-test-datasets)
  + [*Resnet51-S*](#training-resnet51-s)
  + [*AFFACT*](#training-affact)
  + [*Resnet152*](#training-resnet152)
* [Training on multiple GPUs](#training-on-multiple-gpus)
* [References](#references)

## Quick start

1. Clone the repository:
    
    ```bash
    git clone https://github.com/noahch/PyAffact
   ```

2. Create a new Conda environment and install all dependencies:

    ```bash
    # create a new Conda environment with Python 3.7
    conda create -n affact python=3.7
    
    # install all requirements for bob (image processing library)
    # UNIX based system needed
    conda install -c https://www.idiap.ch/software/bob/conda bob.io.image
    conda install -c https://www.idiap.ch/software/bob/conda bob.ip.base
   
    # install pytorch according to your system's specification (e.g. for RTX 3070):
    conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
    
    # install all other dependencies
    pip install -r requirements.txt
   ```    
   
3. Evaluate a pretrained model:
    
    ```bash
    # inside the conda environment, run the following command:
   python py_affact_evaluate.py --config.name=eval/affact     
    ```
    
4. Train a model:
    
    ```bash
    # inside the conda environment, run the following command:
   python py_affact_train.py --config.name=train/affact     
    ```

See `help(py_affact_train)` and `help(py_affact_evaluate)` for usage and implementation details.

## Prerequisites
In order to train a model, download the CelebA dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

# Intro
 - Abstract

    
# Setup
- Install bob with conda
- Clone Repo
- Install
- WandB
- MultiGPU
- CUDA
 
# Training a model
## Resnet51
    - Results
## Affact
    - Result
    
 - config
 - welches file
 - resultat ordner
 - transformer

# Evaluate
## Generate Test datasets

## Evaluate model
- config
- qualitative
- qunatitative