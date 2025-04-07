import os

from decouple import config

DL_DATASET = config('DL_DATASET')

NON_GEOGRAPHIC_ACCEPT_EXTENSION = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")

PLOT_TRAINING = True

BUFFER_TO_INFERENCE = 68 # pixels overlapping between tiles
VALIDATION_SPLIT = 0.10
TILE_SIZE = 256
ORIGINAL_SIZE = 256

MODEL = os.path.join(DL_DATASET, 'artefacts', 'model')

DL_PARAM = {
    'torch': {
            'image_training_folder': os.path.join(DL_DATASET, 'data', 'image', str(ORIGINAL_SIZE), 'train'),
            'annotation_training_folder': os.path.join(DL_DATASET, 'data', 'label', str(ORIGINAL_SIZE), 'train_labels'),
            'image_prediction_folder': os.path.join(DL_DATASET, 'data', 'image', str(ORIGINAL_SIZE), 'test'),
            'mask_prediction_folder': os.path.join(DL_DATASET, 'data', 'label', str(ORIGINAL_SIZE), 'test_labels'),
            'output_checkpoints': os.path.join(DL_DATASET, 'artefacts', 'weights'),
            'save_model_dir': os.path.join(DL_DATASET, 'artefacts', 'model'),
            'save_plot_dir': os.path.join(DL_DATASET, 'artefacts', 'plots'),
            'pretrained_weights': '',
            # 'output_prediction': os.path.join(DL_DATASET, 'artefacts', 'predictions'),
            'output_prediction': os.path.join(DL_DATASET, 'artefacts', 'predictions', str(ORIGINAL_SIZE)),
            'input_size_w': TILE_SIZE,
            'input_size_h': TILE_SIZE,
            'input_size_c': 3,
            'batch_size': 4,
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
