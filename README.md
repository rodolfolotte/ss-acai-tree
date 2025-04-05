# Semantic Segmentation using Deep Learning for Açai trees detection and classification


## Summary
1. [`Model choice and parameters`](#model-choice-and-parameters)   
2. [`Project preparation`](#project-preparation)   
3. [`Prepare your virtual environment`](#prepare-your-virtual-environment)
4. [`Installing `requirements.txt](#installing-requirementstxt)
5. [`Dataset characteristics`](#dataset-characteristics)
6. [`Define `settings.py](#define-settingspy)
8. [`Torch`](#torch)
   1. [`Results`](#results-1)
9. [`Main parameters in `settings.py`](#main-parameters-in-settingspy)
10. [`The hierarchy of folders`](#the-hierarchy-of-folders)
11. [`Overlaping results`](#overlaping-results)
     
   
## Model choice and parameters
The model `deeplabv3+`, referenced [here](https://arxiv.org/abs/1706.05587), was chosen for this solution due to this ability to get small details against rough edges. It combines depthwise separable convolutions (from Xception) (i.e. drastically reduces the number of parameters while maintaining performance) with Atrous Spatial Pyramid Pooling (ASPP) to capture both local and global features effectively, making it one of the most efficient models for segmentation. Besides, it uses the ASPP (Atrous Spatial Pyramid Pooling) convolution, which helps capture multi-scale contextual information without losing resolution.

DeepLabV3+ has shown state-of-the-art performance on popular datasets such as PASCAL VOC, COCO, and Cityscapes. The model has set new records in terms of accuracy, especially for fine-grained semantic segmentation tasks. It is also faster and more memory-efficient than traditional models like U-Net or fully convolutional networks (FCNs).

## Project preparation
The `ss-acai-tree` was developed using Python version 3.9, and **Linux 24.04.2 LTS noble** operational system. 

## Prepare your virtual environment
First of all, check if you have installed the libraries needed:
```
sudo apt-get install python3-env
```
then, in the
```
python -m venv .venv
```
and activates it:
```
source .venv/bin/activate
```
as soon you have it done, you are ready to install the requirements.

## Installing `requirements.txt`
If you do not intent to use GPU, there is no need to install support to it. So, in requirements file, make sure to set `tensorflow-gpu` to only `tensorflow`. If everything is correct, and you **virtualenv** is activated, execute: 
```
pip install -r requirements.txt
```

## Dataset characteristics
REF

## Define `settings.py`
Beyond the parameters set along the process, some of them could be customized by the user. 

This library uses decoupling, which demands you to set up variables that is only presented locally, for instance, the path you want to save something, or the resources of your project. In summary, your environment variables. So, copy a paste the file `.env-example` and rename it to `.env`. Afterwards, just fill out each of the variables content within the file:
```
DL_DATASET = PATH_TO_PARENT_FOLDER_DATASET
```

## Torch
Torch is a framework that tends to be a bit more straightforward, with no need to develop a module with all layers set. Instead, it brings some already built model and the developer has to deal with inputs and outputs (pre and postprocessing). As mentioned in the previous approach with keras, in torch solution was built as such. 

### Results
The results using torch were a bit more promising than keras. The previous approach has a particular and meticulous way to deal with parameters during training and inference. It is true that a couple of modules built for keras approach has to be reviewed, specially for the loss function and the last layer of deeplabv3+, considering a binary classification. 

On the other hands, it is worth to mention that torch is not only more practical approach, but it got better results, as can be seen in `artefacts/predictions`. The following image show the performance of a deeplabv3+ for binary semantic segmentation, using a set of 100 samples, where 90 were used to train, 10 for validation, 100 epochs. The accuracy reached was 97,72%, IoU in 85,04%.
```
...
Epoch 98/100:  100%|██████████████████████████████████████████████████████████████████████████| 23/23 [00:05<00:00,  4.28it/s]
>>>> Epoch 98 | Loss: 0.0934 | IoU: 0.8509 | Accuracy: 0.9781
Epoch 99/100:  100%|██████████████████████████████████████████████████████████████████████████| 23/23 [00:05<00:00,  4.29it/s]
>>>> Epoch 99 | Loss: 0.0978 | IoU: 0.8444 | Accuracy: 0.9758
Epoch 100/100: 100%|██████████████████████████████████████████████████████████████████████████| 23/23 [00:05<00:00,  4.29it/s]
>>>> Epoch 100 | Loss: 0.0944 | IoU: 0.8504 | Accuracy: 0.9772
```

<img src="pics/results-torch.png">

## Main parameters in `settings.py`
The settings will bring all parameters needed. In addition, the following parameters are essential to take a look in every training or inference.
```
PLOT_TRAINING = True
IS_TRAINING = True
IS_PREDICT = True

BUFFER_TO_INFERENCE = 80
VALIDATION_SPLIT = 0.10
TILE_SIZE = 512
ORIGINAL_SIZE = 1200
```

## The hierarchy of folders

The results were organized in `artefacts` folder, with the following hierarchy: 
```
.
├── artefacts
│   ├── model
│   ├── prediction_overlap
│   ├── predictions
│   └── weights
├── data
│   ├── test
│   ├── test_labels
│   ├── train
│   └── train_labels
├── modules
├── pics
```

The folder `output` will store all results processed by the solution presented. Not only predictions, the output also split the results in `model` used, the `weights` after training. The original datasets were placed in the `data` folder.

### Overlaping results
An additional script was made to overlap the results over the original dataset. It is a shell script, that will demand `imagemagick` to be installed in Linux machines. To run the script, three arguments are needed: the original, results, and output folders.
```
./overlap_results.sh
```
> The script was not parametrized. Open and edit the file to customize the respective directories.
