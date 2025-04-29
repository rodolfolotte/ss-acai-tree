import os
import sys
import gc
import torch
import settings
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models.segmentation as models
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataloader import Loader
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from modules import augment


def list_entries(directory):
    """
    Image training and validation loader. List the entries in the folder and its subfolders

    :param directory:
    :return input_img_paths: list
    """
    input_img_paths = sorted([os.path.join(root, name)
                              for root, dirs, files in os.walk(directory)
                              for name in files
                              if (name.endswith(settings.GEOGRAPHIC_ACCEPT_EXTENSION) or
                                  name.endswith(settings.NON_GEOGRAPHIC_ACCEPT_EXTENSION)) and not
                              name.startswith(".")])

    logging.info(">>>> Number of samples: {}".format(len(input_img_paths)))
    return input_img_paths


def compute_metrics(outputs, targets, threshold=0.5):
    with torch.no_grad():
        preds = (torch.sigmoid(outputs) > threshold).float()

        tp = (preds * targets).sum(dim=(1, 2))
        fp = (preds * (1 - targets)).sum(dim=(1, 2))
        fn = ((1 - preds) * targets).sum(dim=(1, 2))

        epsilon = 1e-7
        precision = (tp / (tp + fp + epsilon)).mean().item()
        recall = (tp / (tp + fn + epsilon)).mean().item()
        f1 = (2 * precision * recall / (precision + recall + epsilon))

        intersection = (preds * targets).sum(dim=(1, 2))
        union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - intersection

        # Avoid NaN: If union is zero, set IoU to 1 (perfect match) or 0 (no overlap)
        iou = torch.where(union > 0, intersection / union, torch.tensor(1.0)).mean().item()

        accuracy = (preds == targets).float().mean().item()

    return iou, accuracy, precision, recall, f1


def collate_fn_predict(batch):
    """
    :param batch:
    """
    images, paths = zip(*batch)
    return torch.stack(images, dim=0), list(paths)


def plot_training_history(save_path, train_iou_history, val_iou_history, train_acc_history, val_acc_history,
                          train_prec_history, val_prec_history, train_rec_history, val_rec_history,
                          train_f1_history, val_f1_history):
    """

    """
    epochs = range(1, len(train_iou_history) + 1)
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_iou_history, 'b-o', label='Train IoU')
    plt.plot(epochs, val_iou_history, 'r--o', label='Val IoU')
    plt.title("IoU over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_acc_history, 'b-o', label='Train Acc')
    plt.plot(epochs, val_acc_history, 'r--o', label='Val Acc')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_prec_history, 'b-o', label='Train Prec')
    plt.plot(epochs, val_prec_history, 'r--o', label='Val Prec')
    plt.title("Precision over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, train_rec_history, 'b-o', label='Train Recall')
    plt.plot(epochs, val_rec_history, 'r--o', label='Val Recall')
    plt.title("Recall over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_f1_history, 'b-o', label='Train F1')
    plt.plot(epochs, val_f1_history, 'r--o', label='Val F1')
    plt.title("F1-score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f">>>> Training plot saved to {save_path}")


def delete_low_white_images(image_paths, threshold=0.15):
    """
    Deletes images from the list if they contain less than a threshold percentage of white pixels.

    Parameters:
    - image_paths (list of str): List of paths to binary images (white = 255, black = 0).
    - threshold (float): Minimum required proportion of white pixels to keep the image (0 to 1).
    """
    for path in image_paths:
        image = Image.open(path).convert("L")
        image = np.array(image)

        if image is None:
            print(f"Warning: Could not read image at {path}")
            continue

        total_pixels = image.size
        white_pixels = np.count_nonzero(image == 255)

        white_ratio = white_pixels / total_pixels

        if white_ratio < threshold:
            try:
                os.remove(path)
                print(f"Deleted {path} (white ratio: {white_ratio:.2%})")
            except Exception as e:
                print(f"Error deleting {path}: {e}")


def remove_already_augmented(train_images, train_labels):
    """
    """
    train_images_filtered = list_entries(train_images)
    train_labels_filtered = list_entries(train_labels)

    train_images_filtered = [file_path for file_path in train_images_filtered if
                             'aug' not in os.path.basename(file_path)]
    train_labels_filtered = [file_path for file_path in train_labels_filtered if
                             'aug' not in os.path.basename(file_path)]
    return train_images_filtered, train_labels_filtered


def initialize(load_param, augment_data, is_training, is_predicting):
    """

    :param load_param:
    :param augment_data:
    :param is_training:
    :param is_predicting:
    :return:
    """
    transform = T.Compose([
        # T.Resize((settings.ORIGINAL_SIZE, settings.ORIGINAL_SIZE)),
        # T.CenterCrop(settings.ORIGINAL_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = load_param['image_training_folder']
    mask_dir = load_param['annotation_training_folder']

    timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M")
    model_path = os.path.join(load_param['save_model_dir'], "deeplabv3-" + timestamp + ".pth")
    plot_filepath = os.path.join(load_param['save_plot_dir'], "deeplabv3-" + timestamp + ".png")

    model = models.deeplabv3_resnet50(pretrained=True)
    # model = models.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model.to(device)

    if eval(augment_data):
        logging.info(">> Augmenting entries...")
        img_size = (load_param['input_size_w'], load_param['input_size_h'])

        train_images_paths, train_labels_paths = remove_already_augmented(image_dir, mask_dir)

        augmentor = augment.Augment(img_size, train_images_paths, train_labels_paths)
        augmentor.augment()

    if eval(is_training):
        logging.info(">> Loading input datasets...")

        dataset = Loader(image_dir, mask_dir, transform)

        val_size = int(settings.VALIDATION_SPLIT * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=load_param['batch_size_training'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=load_param['batch_size_training'], shuffle=False)

        train_iou_history, train_acc_history, train_prec_history, train_rec_history, train_f1_history = [], [], [], [], []
        val_iou_history, val_acc_history, val_prec_history, val_rec_history, val_f1_history = [], [], [], [], []

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=load_param['learning_rate'])

        num_epochs = load_param['epochs']
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_iou = 0.0
            total_accuracy = 0.0
            total_prec = 0.0
            total_f1 = 0.0
            total_rec = 0.0

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)["out"]
                loss = criterion(outputs.squeeze(1), masks.squeeze(1))

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                iou, acc, prec, rec, f1 = compute_metrics(outputs, masks)
                total_iou += iou
                total_accuracy += acc
                total_prec += prec
                total_rec += rec
                total_f1 += f1

            avg_loss = running_loss / len(train_loader)
            avg_iou = total_iou / len(train_loader)
            avg_acc = total_accuracy / len(train_loader)
            avg_prec = total_prec / len(train_loader)
            avg_rec = total_rec / len(train_loader)
            avg_f1 = total_f1 / len(train_loader)
            train_iou_history.append(avg_iou)
            train_acc_history.append(avg_acc)
            train_prec_history.append(avg_prec)
            train_rec_history.append(avg_rec)
            train_f1_history.append(avg_f1)

            logging.info(f">>>> Epoch {epoch + 1} | Train Loss: {avg_loss:.4f} | "
                         f"Train IoU: {avg_iou:.4f} | Train Acc: {avg_acc:.4f} | "
                         f"Train Precision: {avg_prec:.4f} | Train Recall: {avg_rec:.4f} | "
                         f"Train F1-score: {avg_f1:.4f}")

            model.eval()
            val_total_iou = 0.0
            val_total_accuracy = 0.0
            val_total_prec = 0.0
            val_total_rec = 0.0
            val_total_f1 = 0.0

            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]"):
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)["out"]

                    iou, acc, prec, rec, f1 = compute_metrics(outputs, masks)
                    val_total_iou += iou
                    val_total_accuracy += acc
                    val_total_prec += prec
                    val_total_rec += rec
                    val_total_f1 += f1

            val_avg_iou = val_total_iou / len(val_loader)
            val_avg_acc = val_total_accuracy / len(val_loader)
            val_avg_prec = val_total_prec / len(val_loader)
            val_avg_rec = val_total_rec / len(val_loader)
            val_avg_f1 = val_total_f1 / len(val_loader)
            val_iou_history.append(val_avg_iou)
            val_acc_history.append(val_avg_acc)
            val_prec_history.append(val_avg_prec)
            val_rec_history.append(val_avg_rec)
            val_f1_history.append(val_avg_f1)

            logging.info(f">>>> Epoch {epoch + 1} | Val IoU: {val_avg_iou:.4f} | "
                         f"Val Acc: {val_avg_acc:.4f} | Val Precision: {val_avg_prec:.4f} | "
                         f"Val Recall: {val_avg_rec:.4f} | Val F1-score: {val_avg_f1:.4f}")

        logging.info(">>>>>> Model built. Saving model in {}...".format(model_path))
        torch.save(model.state_dict(), model_path)

        if settings.PLOT_TRAINING:
            plot_training_history(plot_filepath, train_iou_history, val_iou_history, train_acc_history, val_acc_history,
                                  train_prec_history, val_prec_history, train_rec_history, val_rec_history,
                                  train_f1_history, val_f1_history)

    if eval(is_predicting):
        logging.info(">> Performing prediction...")

        test_dataset = Loader(load_param['image_prediction_folder'], None, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=load_param['batch_size_prediction'], shuffle=False, collate_fn=collate_fn_predict)

        torch.cuda.empty_cache()
        gc.collect()

        if load_param['pretrained_weights'] != '':
            model = models.deeplabv3_resnet50(pretrained=True)
            # model = models.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
            model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
            model.to(device)

            model_filename = os.path.join(load_param['save_model_dir'], load_param['pretrained_weights'])
            model.load_state_dict(torch.load(model_filename, weights_only=True, map_location=device), strict=False)
            model.eval()

        for batch_images, batch_paths in tqdm(test_loader, desc="Predicting"):
            batch_images = batch_images.to(device)

            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = model(batch_images)["out"]
                preds = torch.sigmoid(outputs).detach().cpu().squeeze(1).numpy()

            for img_path, pred in zip(batch_paths, preds):
                filename = os.path.basename(img_path)
                pred_image = (pred > 0.5).astype('uint8') * 255

                os.makedirs(load_param['output_prediction'], exist_ok=True)

                Image.fromarray(pred_image).save(os.path.join(load_param['output_prediction'], filename))
