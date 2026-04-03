# Semantic Segmentation using Deep Learning for Açai trees detection and classification


## Summary
1. [`Model choice and parameters`](#model-choice-and-parameters)   
2. [`Project preparation`](#project-preparation)   
3. [`Prepare your virtual environment`](#prepare-your-virtual-environment)
4. [`Installing requirements`](#installing-requirements)
5. [`Dataset characteristics`](#dataset-characteristics)
6. [`Define settings.py`](#define-settingspy)
7. [`Torch`](#torch)
   1. [`Results`](#results-1)
8. [`Main parameters in settings.py`](#main-parameters-in-settingspy)
9. [`Running the module`](#running-the-module)
10. [`The hierarchy of folders`](#the-hierarchy-of-folders)
11. [`Overlaping results`](#overlaping-results)
     
   
## Model choice and parameters
The model `deeplabv3+`, referenced [here](https://arxiv.org/abs/1706.05587), was chosen for this solution due to its ability to get small details against rough edges. It effectively combines Atrous Spatial Pyramid Pooling (ASPP) to capture both local and global features, making it one of the most efficient models for segmentation. In this project, DeepLabV3+ is configured to use either **ResNet50** or **MobileNetV3Large** as its backbone using `torchvision` pre-trained models.

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

## Installing requirements
If you do not intent to use GPU, there is no need to install support to it. So, in your environment, make sure to adjust Torch packages accordingly. If everything is correct, and your **virtualenv** is activated, execute: 
```
pip install -r requirements_linux.txt
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
Torch is a framework that tends to be a bit more straightforward, with no need to develop a module with all layers set. Instead, it brings some already built models (like `torchvision`'s implementation of DeepLabV3+) and the developer has to deal with inputs and outputs (pre and postprocessing).

In `modules/initialize.py`, the Torch solution provides an automated pipeline that will transparently split the raw training data into `val` and `test` directories based on `VALIDATION_SPLIT` and `TEST_SPLIT` parameters. It also supports on-the-fly customized data augmentation natively and implements Early Stopping to save the best model weights dynamically.

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
The settings will bring all parameters needed. In addition, the following parameters are essential to review before proceeding:
```python
MODEL_NAME = 'resnet50'  # Options: 'resnet50' or 'mobilenet'
PLOT_TRAINING = True

BUFFER_TO_INFERENCE = 80
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.05
TILE_SIZE = 256
ORIGINAL_TILE_SIZE = 256
ORIGINAL_SCENE_SIZE = 2048
```

## Running the module
Unlike some parameters, actions like training, predicting, and data argumentation are handled directly by parameter flags in `main.py` instead of the settings file. You can easily execute combinations of operations:

```bash
python main.py -augment False -train True -validate False -predict True -verbose True
```

## The hierarchy of folders

The results are fully organized in the `artefacts` folder, avoiding floating files. The general structure looks like: 
```text
.
├── artefacts
│   ├── model
│   ├── plots
│   ├── predictions
│   └── weights
├── data
├── modules
├── scripts
└── pics
```

The folder `artefacts` will store all results processed by the solution presented. Not only predictions, the output also separates the `model` used, the `weights` saved after training (including best checkpoints), and the training `plots`. The original raw datasets are placed in the `data` folder.

## Additional scripts

### Crop Original Images
This script (`scripts/crop_original_image.sh`) uses `imagemagick` to divide a large source `.tif` image into smaller square patches (tiles) based on a specified tile size and overlap amount. It gracefully handles borders by padding with a black background if necessary.
```bash
./scripts/crop_original_image.sh <input_folder> <output_folder> <size> <overlap>
```

### Regularize Validation and Train Folders
A helper script (`scripts/regularize_val_train_folder.sh`) designed to clean up and sync training vs. validation split directories. If you manually place images into the validation folder, this script parses those base file names to delete any exact matches or augmented versions in the `train` folder (preventing data leakages). Concurrently, it moves corresponding label masks from `labels/train` to the `labels/val` directory.
```bash
./scripts/regularize_val_train_folder.sh <image_train_dir> <image_val_dir> <labels_train_dir>
```

### Overlaping results
An additional script was made to overlap the results over the original dataset. It is a shell script, that will demand `imagemagick` to be installed in Linux machines. The script is fully parameterized and expects three arguments: the original images folder, predictions folder, and the output folder.
```bash
./scripts/overlap_results.sh <folder_original_images> <folder_predictions> <output_folder>
```
