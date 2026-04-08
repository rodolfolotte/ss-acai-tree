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
pip install -r requirements.txt
```

## Dataset characteristics

### How to prepare input dataset
The geospatial data pipeline requires structured input directories holding raw tiles and corresponding categorical masks. 

**When:** Before triggering the `main.py` pipeline.

**How:** 
1. Place raw optical/multispectral geographic tiles (e.g., `.tif`, `.png`, `.jpg`) into the designated images directory (`data/image/256/train/`). 
2. Place the corresponding boolean semantic masks under the labels directory (`data/label/256/train/`), matching the exact filenames of their image counterparts.
3. Masks must implicitly map to the target taxonomy configuration (background as `0`, Açaí pixels typically corresponding spatially to categorical references or logic defined in settings).

**Why:** Structuring data identically between spatial features and ground-truth allows PyTorch `DataLoader` objects to index targets dynamically. Additionally, the pre-processing layer dynamically evaluates terrain representations, systematically filtering out and rejecting image patches showing insufficient visible geospatial geography (defined as a white-pixel ratio `< 0.15`), maintaining optimal algorithmic signal-to-noise ratio.

### Overlapping tiles
Tiling converts massive geospatial orthomosaics (e.g., `2048x2048`) into memory-manageable dimensional grids (e.g., `256x256`).

**When:** This occurs strictly during *data preprocessing/creation process* (using external scripts like `crop_original_image.sh`), prior to model training.

**How & Why:** 
When slicing geographic matrices uniformly, morphological targets (tree canopies) will inevitably land directly on the edge boundaries. If chopped abruptly without care, the model cannot reliably learn the localized shape of the object. 

To combat this, an **80-pixel buffer overlap** (customizable) is mathematically incorporated while sliding the cropping window. 

*Crucial Note on Inference*: Because the DeepLabV3+ architecture is fundamentally **Fully Convolutional** (FCN) and invariant to total spatial dimensionality, the model scales its operations natively over variable-dimension imagery during production inference. Due to this architectural feature, computational **re-combination of overlapping patches is entirely unnecessary and omitted** post inference!

### Spliting train, val and test data

Splitting physically partitions your dataset so the neural network optimizes weights against one main data fraction (*Train*) while being evaluated impartially against separate, untouched fractions (*Validation/Test*) to track training progress objectively without overfitting to memory.

**When:** Handled dynamically during the early execution phases of `initialize.py`.

**How:** 
Using parameters mapped internally inside `settings.py` (e.g., `VALIDATION_SPLIT = 0.10` and `TEST_SPLIT = 0.00`), the `create_train_val_test_split` orchestration engine randomly samples un-augmented base tiles within the primary `/train` directories. It physically moves the appropriate volumes of image-mask pairs out of `/train` and into automatically generated neighboring `/val` and `/test` folders.

**Why:** Executing dataset severance explicitly *before* the application of stochastic data augmentation procedures (rotations, blurs, scaling) holds critical importance. This sequence mathematically guarantees that intensely morphed copies of a spatial image cannot accidentally contaminate the validation arrays, effectively establishing zero-trust and preventing Data Leakage.

### Expected outputs
The expected output of the training process is a set of model weights (including best checkpoints) stored in `artefacts/weights`, training plots in `artefacts/plots`, and predictions in `artefacts/predictions`. The predictions will be saved as images, where the predicted masks can be overlaid on the original images for visual assessment. The training plots will include metrics such as loss, IoU, and accuracy over epochs, allowing you to evaluate the training process and identify potential issues like overfitting or underfitting.

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

<img src="pics/results-torch.png" alt="Training results">

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
Unlike some parameters, actions like training, predicting, and data argumentation are handled directly by parameter flags in `main.py` instead of the `settings.py` file. You can easily execute combinations of operations:

```bash
python main.py -augment False -train True -validate False -predict True -verbose True
```


## Tested software versions (this repo state)
- Python: tested with Python 3.9 (the project was developed using Python 3.9; `main.py` requires Python 3+).
- Core libraries (exact pinned versions are in `requirements.txt`):
  - torch==2.6.0
  - torchvision==0.21.0
  - numpy==1.26.4
  - pillow==12.2.0
  - torchvision models (DeepLabV3+ backbones via `torchvision`)

Note: `requirements.txt` contains the full set of pinned packages used during development. Use that file to reproduce the exact environment.


## Typical install time (guidance)
- CPU-only install (common desktop, 4 cores, 16 GB RAM, reasonable broadband): expect roughly 5–20 minutes to install the packages in `requirements.txt` with `pip install -r requirements.txt`. Network speed and pip caches strongly affect this.
- GPU-enabled install (downloads for CUDA toolkits / large wheels, and additional NVIDIA packages present in `requirements.txt`): plan for 10–45 minutes depending on your connection and whether you need to install CUDA/system drivers. Installing PyTorch with the matching CUDA wheels according to the instructions at https://pytorch.org is recommended and may be faster and more reliable than installing everything from `requirements.txt` at once.

Tips:
- Use a virtual environment before installing.
- If you have an NVIDIA GPU, prefer to install the correct PyTorch+CUDA wheel from PyTorch's site for your CUDA version instead of relying on a single giant `pip install -r requirements.txt` to pull many GPU-specific artifacts.


## License
This project is licensed under the MIT License. See `LICENSE` for the full text. In short: you are free to use, copy, modify, and distribute the software, provided you preserve the copyright and license notices.


## Pseudocode / Pipeline overview
A concise high-level view of how the repository runs (follow `main.py` -> `modules.initialize.initialize` -> `Loader`):

1. Prepare environment variable `DL_DATASET` (or `.env`) pointing to your dataset root.
2. Run `main.py` with flags to select augmentation / training / validation / prediction.
3. `main.py` reads `settings.DL_PARAM['torch']` into `load_param` and delegates to `modules.initialize.initialize(load_param, augment_data, is_training, is_validating, is_predicting)`.
4. `initialize` builds the DeepLabV3+ model (ResNet50 or MobileNet based on `settings.MODEL_NAME`) and adapts the final layer for binary segmentation.
5. If `-augment True`: augmentation pipeline runs and writes augmented images to the training folders.
6. If `-train True`:
   - `create_train_val_test_split` ensures `train/`, `val/`, and `test/` folders (physically moves files if needed)
   - Train loop runs with BCEWithLogitsLoss and Adam optimizer; early stopping saves best checkpoints in `artefacts/weights`.
7. If `-validate True`: load `pretrained_weights` (if configured) and compute dataset-wide IoU/precision/recall; save metrics and plots to `artefacts/plots`.
8. If `-predict True`:
   - A `Loader` reads images from `image_prediction_folder`; model runs inference (batch size configured by `batch_size_prediction`, default 1) and predictions are saved under `output_prediction`.

Key settings to review: `DL_DATASET` (root), `MODEL_NAME`, `DL_PARAM['torch']['pretrained_weights']`, `input_size_w/h`, `batch_size_prediction`, and `output_prediction` (see `settings.py`).


## Demo — run the provided `example/` data (quick start)
This quick demo runs the prediction pipeline on the small `example/` dataset included in the repository and saves outputs to the example `artefacts` folder.

1. Set `DL_DATASET` to the absolute path of the `example/` folder. You can either export it in your shell or create a `.env` file with `DL_DATASET` (the code uses `python-decouple` to read it):

```bash
# from the repo root
export DL_DATASET="$(pwd)/example"
# or create .env in the repo root with the line:
# DL_DATASET=/absolute/path/to/example
```

2. Prepare example artefacts folder and copy the provided example model into `artefacts/weights` so the `pretrained_weights` entry in `settings.py` can be found:

```bash
mkdir -p example/artefacts/weights
cp example/model/*.pth example/artefacts/weights/ || true
```

(If you prefer, edit `settings.py` to point `pretrained_weights` to the exact filename placed under `example/artefacts/weights`.)

3. Run prediction only (no training) over the example images and enable verbose logging:

```bash
python main.py -augment False -train False -validate False -predict True -verbose True
```

4. Where outputs land:
- Prediction masks: `example/artefacts/predictions/<MODEL_NAME>_w_aug/` (see `settings.DL_PARAM['torch']['output_prediction']`).
- Logs: `logging.log` (if run with `-verbose True`).
- If predictions are not created, check `DL_DATASET` and ensure `example/artefacts/weights/` contains the expected `.pth` file referenced in `settings.py`.

Estimated inference times (per 256×256 tile) — rough guidance (measure on your machine for accurate numbers):
- GPU (NVIDIA mid-range e.g., RTX 2060/3060): ~10–100 ms per tile.
- High-end GPU: ~5–30 ms per tile.
- CPU (4 cores, no SIMD/CUDA): ~0.5–3.0 seconds per tile.

These are approximate: real performance depends on model backbone (`resnet50` is heavier than `mobilenet`), device, CPU/GPU load, and whether automatic mixed precision is effectively used. The project uses batch size 1 for prediction by default (`batch_size_prediction`), which favors memory-constrained inference but is less throughput-optimal.

How to measure precisely on your machine:
- Wrap the run with the `time` command, e.g.:

```bash
time python main.py -augment False -train False -validate False -predict True -verbose False
```

- Or, measure a single forward pass in an interactive Python snippet that loads the model and times torch inference using `torch.cuda.synchronize()` when measuring GPU time.

