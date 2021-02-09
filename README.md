# PyAffact
PyTorch implementation of the AFFACT Paper

Facial attribute classification tasks have many potential real-world applications across different areas (e.g. surveillance, entertainment, medical treatment). To make classification models more stable toward misaligned images, a data augmentation technique called alignment-free facial attribute classification technique (AFFACT) was introduced by Günther et al. [[1]](#1). Since their approach was originally implemented using the nowadays outdated Caffe framework, the goal of this master project is to reimplement this technique using the more popular PyTorch framework. Our reimplementation performs on the same level as the original AFFACT model. We further introduce the possibility to perform hyperparameter optimization on existing models and establish an extended network trained on an additional data augmentation operation. 


## Table of contents

* [Table of contents](#table-of-contents)
* [Quick start](#quick-start)
  + [*Dataset*](#dataset)
  + [*Configuration files*](#configuration-files)
  + [*WandB*](#wandb)
* [Pretrained models](#pretrained-models)
* [Training a model](#training-a-model)
  + [*Resnet-51-S*](#training-resnet-51-s)
  + [*AFFACT*](#training-affact)
  + [*AFFACT-Ext*](#training-AFFACT-Ext)
* [Evaluation of a model](#evaluation-of-a-model)
  + [*Generate test datasets*](#generate-test-datasets)
  + [*Resnet-51-S*](#training-resnet-51-s)
  + [*AFFACT*](#training-affact)
  + [*AFFACT-Ext*](#training-AFFACT-Ext)
* [Training on multiple GPUs](#training-on-multiple-gpus)
* [Run a Hyperparameter Optimization](#run-a-hyperparameter-optimization)
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
   
    # activate environment
    conda activate affact

    # install all requirements for bob (image processing library)
    # UNIX based system needed
    conda install -c https://www.idiap.ch/software/bob/conda bob.io.image
    conda install -c https://www.idiap.ch/software/bob/conda bob.ip.base

    # install PyTorch according to your system's specification 
    # have a look at https://pytorch.org/ as the installation command may vary depending on your OS and your version of CUDA

    # install all other dependencies
    pip install -r requirements.txt
   ```    

3. Adjust the dataset path in the configuration files located in config/train and config/eval 

4. Evaluate a pretrained model:

    ```bash
    # inside the conda environment, run the following command:
   python py_affact_evaluate.py --config.name=eval/affact     
    ```

5. Train a model:

    ```bash
    # inside the conda environment, run the following command:
   python py_affact_train.py --config.name=train/affact     
    ```

### Dataset
In order to train a model, download the CelebA dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

### Configuration Files
There are the following 3 types of configuration YAML files. The first configuration file is located in the config/train folder. This configuration file covers all hyperparameters needed to train a model.
Furthermore, there is another configuration file with different parameters in the config/eval folder, which covers all parameters to evaluate a model.
Finally, the configuration file to create the different test sets is situated in the config/dataset folder.
The configuration files are written in the popular YAML format and fully documented.
For reproducibility purposes, there is a configuration file for the ResNet-51-S, AFFACT and AFFACT-Ext in the config/train and in the config/evaluation folder respectively.

For example, the fully-documented configuration YAML file to train the AFFACT model can be found [here](config/train/affact.yml).

### WandB
WandB (Weights and Biases) is a popular tool to keep track of machine learning experiments [[4]](#4).
In our project we allow the use of WandB in our training configuration file.
When enabled, each experiment creates 1 report with a dedicated URL link. This report then monitors the training history and every epoch the plots for the current training and validation loss/accuracy is automatically updated. Moreover, the system metrics such as memory, CPU, GPU usage are recorded and updated on the fly.

The installation guide can be found [here](https://docs.wandb.ai/quickstart).

## Pretrained models
The pretrained models described in the AFFACT paper as well as our own built AFFACT-Ext model are available for download here:

|Model name|
| :- |
|[ResNet-51-S](https://drive.google.com/file/d/1EMk6fAtLLmzNEvlh5B604MGvwDMfxX16/view) |
|[AFFACT](https://drive.google.com/file/d/1D7WVDuDZ49Tl2B9U4077F8uUgUeuwL7A/view) |
|[AFFACT-Ext](https://drive.google.com/file/d/1L5hE3kkaT35oJgmf_FUuOrv7kKVQTvFd/view) |

## Training a model
To simplify training of the models in the AFFACT paper, we created a configuration file for each model in the AFFACT paper. Additionally, we added a configuration file our own model (AFFACT-Ext) which is described in our report. The commands to train the different models are listed below.

### ResNet-51-S
Run the following command to train the ResNet-51-S model:

 ```bash
 # inside the conda environment, run the following command:
 python py_affact_train.py --config.name=train/resnet51_s     
 ```

### AFFACT
Run the following command to train the AFFACT model:

 ```bash
 # inside the conda environment, run the following command:
 python py_affact_train.py --config.name=train/affact     
 ```

### AFFACT-Ext
Run the following command to train the AFFACT-Ext model:

 ```bash
 # inside the conda environment, run the following command:
 python py_affact_train.py --config.name=train/affact_ext     
 ```


## Evaluation of a model
We evaluate each model quantitavely and qualitatively. In the quantitative analysis we analyze the accuracy of the model on 4 transformations of the test set (A, C, D, T) described in our report for each attribute and calculate the overall accuracy.
In the qualitative analysis we look at 6 samples of images and evaluate the prediction quality of the network.
All the results are saved in a result folder, marked by the timestamp and specific chosen model.

### Generate Test datasets
Generate 4 different test sets (A, D, C, T) based on the test partition of the CelebA dataset. 
The configuration files in the config/dataset folder control the transformation of the test set images. 

 ```bash
 # inside the conda environment, run the following command:
 python generate_test_datasets.py     
 ```

### Resnet-51-S
Run the following command to evaluate the ResNet-51-S model:

 ```bash
 # inside the conda environment, run the following command:
 python py_affact_evaluate.py --config.name=eval/resnet51_s     
 ```

### AFFACT
Run the following command to evaluate the AFFACT model:

 ```bash
 # inside the conda environment, run the following command:
 python py_affact_evaluate.py --config.name=eval/affact     
 ```

### AFFACT-Ext
Run the following command to evaluate the AFFACT-Ext model:

 ```bash
 # inside the conda environment, run the following command:
 python py_affact_evaluate.py --config.name=eval/affact_ext     
 ```

## Training on multiple GPUs
It is possible to add several cuda IDs separated by comma in a training configuration file or by using a flag in order to train the model on multiple GPUs. For example, run the following command to train the AFFACT model on cuda:0 and cuda:1.

```bash
# inside the conda environment, run the following command:
python py_affact_evaluate.py --config.name=eval/affact --config.basic.cuda_device_name=cuda:0,cuda:1     
```

## Run a Hyperparameter Optimization
If you want to explore hyperparameters you can run a [Sweep using WandB](https://docs.wandb.com/sweeps) over a range of values.

Edit the example sweep configuration files located in config/train/hyperopt to adjust parameter bounds.

Create the sweep:
- `wandb sweep config/train/hyperopt/affact_hyperopt.yaml`

Start Sweep:
- `wandb agent SWEEP_ID  # SWEEP_ID is from the create sweep command above`


## References
<a id="1">[1]</a> [AFFACT - Alignment-Free Facial Attribute Classification Technique](https://arxiv.org/pdf/1611.06158) \
<a id="2">[2]</a> [Reimplementing AFFACT](https://drive.google.com/file/d/1KLzqxLs9m_XbakMimGYGTteRLiGVzZFv/view)
<a id="3">[3]</a> [PyTorch](https://pytorch.org) \
<a id="4">[4]</a> [Wandb](http://wandb.ai)
