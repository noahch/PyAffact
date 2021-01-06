# PyAffact
PyTorch implementation of the Affact Paper


Abstract

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
  + [*Resnet-152*](#training-resnet-152)
* [Evaluation of a model](#evaluation-of-a-model)
  + [*Generate test datasets*](#generate-test-datasets)
  + [*Resnet-51-S*](#training-resnet-51-s)
  + [*AFFACT*](#training-affact)
  + [*Resnet-152*](#training-resnet-152)
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

### Dataset
In order to train a model, download the CelebA dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

### Configuration Files
There are the following 3 types of configuration YAML files. The first configuration file is located in the config/train folder. This configuration file covers all hyperparameters needed to train a model.
Furthermore, there is another configuration file with different parameters in the config/eval folder, which covers all parameters to evaluate a model.
Finally, the configuration file to create the different test sets is situated in the config/dataset folder.
The configuration files are written in the popular YAML format and fully documented.
For reproducibility purposes, there is a configuration file for the ResNet-51-S, AFFACT and ResNet-152 in the config/train and in the config/evaluation folder respectively.

For example, the fully-documented configuration YAML file to train the AFFACT model can be found [here](config/train/affact.yml).

### WandB
WandB (Weights and Biases) is a popular tool to keep track of machine learning experiments \cite{wandb}.
In our project we allow the use of WandB in our training configuration file.
When enabled, each experiment creates 1 report with a dedicated URL link. This report then monitors the training history and every epoch the plots for the current training and validation loss/accuracy is automatically updated. Moreover, the system metrics such as memory, CPU, GPU usage are recorded and updated on the fly.

## Pretrained models
The pretrained models described in the AFFACT paper as well as own built ResNet-152 model achieved the following results:

|Model name|Accuracy on test set|
| :- | :-: |
|[ResNet-51-S](xxx) (xxxMB)|0.xxx|
|[AFFACT](xxx) (xxxMB)|0.xxx|
|[ResNet-152](xxx) (xxxMB)|0.xxx|

## Training a model
To simplify training of the models in the AFFACT paper, we created a configuration file for each model in the AFFACT paper. Additionally, we added a configuration file our own model (ResNet-152) which is described in our report. The commands to train the different models are listed below.

### Resnet-51-S
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

### ResNet-152
Run the following command to train the ResNet-152 model:

 ```bash
 # inside the conda environment, run the following command:
 python py_affact_train.py --config.name=train/resnet152     
 ```


## Evaluation of a model
We evaluate each model quantitavely and qualitatively. In the quantitative analysis we analyze the accuracy of the model on 5 transformations of the test set (AA, AM, AZ, C, T) described in our report for each attribute and calculate the overall accuracy.
In the qualitative analysis we look at 6 samples of images and evaluate the prediction quality of the network.
All the results are saved in a result folder, marked by the timestamp and specific chosen model.

### Generate Test datasets
Generate 5 transformations (AA, AM, AZ, C, T) on the test partition of the CelebA dataset:

 ```bash
 # inside the conda environment, run the following command:
 python generate_test_datasets.py     
 ```

### Resnet-51-S
Run the following command to evaluate the ResNet-51-S model:

 ```bash
 # inside the conda environment, run the following command:
 python py_affact_evaluate.py --config.name=train/resnet51_s     
 ```

### AFFACT
Run the following command to evaluate the AFFACT model:

 ```bash
 # inside the conda environment, run the following command:
 python py_affact_evaluate.py --config.name=train/affact     
 ```

### ResNet-152
Run the following command to evaluate the ResNet-152 model:

 ```bash
 # inside the conda environment, run the following command:
 python py_affact_evaluate.py --config.name=train/resnet152     
 ```

## Training on multiple GPUs
It is possible to add several cuda IDs seperated by comma in a training configuration file or by using a flag in order to train the model on multiple GPUs. For example, run the following command to train the AFFACT model on cuda:0 and cuda:1.

```bash
# inside the conda environment, run the following command:
python py_affact_evaluate.py --config.name=train/affact --config.basic.cuda_device_name=cuda:0,cuda:1     
```

## References
AFFACT \
pytorch \
our report
