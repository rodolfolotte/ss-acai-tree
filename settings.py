import os

from decouple import config

DL_DATASET = config('DL_DATASET')

NON_GEOGRAPHIC_ACCEPT_EXTENSION = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")

PLOT_TRAINING = True

BUFFER_TO_INFERENCE = 68 # pixels overlapping between tiles
VALIDATION_SPLIT = 0.10
TILE_SIZE = 128
ORIGINAL_SIZE = 128

MODEL = os.path.join(DL_DATASET, 'artefacts', 'model')

DL_PARAM = {
    'torch': {
            'image_training_folder': os.path.join(DL_DATASET, 'data', 'train_128'),
            'annotation_training_folder': os.path.join(DL_DATASET, 'data', 'train_128_labels'),
            'image_prediction_folder': os.path.join(DL_DATASET, 'data', 'test_128'),
            'mask_prediction_folder': os.path.join(DL_DATASET, 'data', 'test_128_labels'),
            'output_checkpoints': os.path.join(DL_DATASET, 'artefacts', 'weights'),
            'save_model_dir': os.path.join(DL_DATASET, 'artefacts', 'model'),
            'pretrained_weights': 'deeplabv3-05-Apr-2025-16-57.pth',
            # 'output_prediction': os.path.join(DL_DATASET, 'artefacts', 'predictions'),
            'output_prediction': '/media/rodolfo/data/sacha/',
            'input_size_w': TILE_SIZE,
            'input_size_h': TILE_SIZE,
            'input_size_c': 3,
            'batch_size': 16,
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
