import os

from decouple import config

DL_DATASET = config('DL_DATASET')

GEOGRAPHIC_ACCEPT_EXTENSION = (".TIF", ".tif", ".tiff", ".TIFF")
NON_GEOGRAPHIC_ACCEPT_EXTENSION = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")

PLOT_TRAINING = True
LABEL_TYPE = 'classid'

BUFFER_TO_INFERENCE = 80 # pixels overlapping between tiles
VALIDATION_SPLIT = 0.10
TILE_SIZE = 256
ORIGINAL_TILE_SIZE = 256
ORIGINAL_SCENE_SIZE = 2048
MODEL = os.path.join(DL_DATASET, 'artefacts', 'model')

AUGMENTATION_TRANSFORMS = ['rotation', 'blured', 'color-1', 'color-2', 'resize-1', 'resize-2']

DL_PARAM = {
    'torch': {
            'image_training_folder': os.path.join(DL_DATASET, 'data', 'image', str(ORIGINAL_TILE_SIZE), 'train'),
            'annotation_training_folder': os.path.join(DL_DATASET, 'data', 'label', str(ORIGINAL_TILE_SIZE), 'train_labels'),
            'image_prediction_folder': os.path.join(DL_DATASET, 'data', 'image', 'original', 'test'),
            'mask_prediction_folder': os.path.join(DL_DATASET, 'data', 'label', str(ORIGINAL_TILE_SIZE), 'test_labels'),
            'output_checkpoints': os.path.join(DL_DATASET, 'artefacts', 'weights'),
            'save_model_dir': os.path.join(DL_DATASET, 'artefacts', 'model'),
            'save_plot_dir': os.path.join(DL_DATASET, 'artefacts', 'plots'),
            'pretrained_weights': 'deeplabv3-28-Apr-2025-11-41_resnet_w_aug_256.pth',
            'output_prediction': os.path.join(DL_DATASET, 'artefacts', 'predictions', "resnet_w_aug_2"),
            'input_size_w': TILE_SIZE,
            'input_size_h': TILE_SIZE,
            'input_size_c': 3,
            'batch_size_training': 8,
            'batch_size_prediction': 1,
            'learning_rate': 0.00001,
            'patience': 20,
            'epochs': 100,
            'classes': {
                    "other": [0, 0, 0],
                    "acai": [102, 153, 0],
            },
            'color_classes': {0: [0, 0, 0], 1: [102, 153, 0]},
    }
}
