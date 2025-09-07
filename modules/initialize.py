import os
import sys
import gc
import torch
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights
)

import settings
import random
import shutil
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
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, average_precision_score


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


def compute_metrics(outputs, masks, threshold=0.5):
    """Binary segmentation metrics, ignoring background."""
    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    preds = (probs > threshold).astype(np.uint8)
    gts = masks.cpu().numpy().astype(np.uint8)

    # Flatten per batch
    preds = preds.reshape(-1)
    gts = gts.reshape(-1)

    tp = np.sum((preds == 1) & (gts == 1))
    fp = np.sum((preds == 1) & (gts == 0))
    fn = np.sum((preds == 0) & (gts == 1))
    tn = np.sum((preds == 0) & (gts == 0))

    if tp + fp == 0:
        prec = 0.0
    else:
        prec = tp / (tp + fp + 1e-8)

    iou = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)



    return iou, acc, prec, rec, f1, preds, gts


# def evaluate_and_plot(all_preds, all_gts, per_image_iou):
#     # Confusion Matrix (only foreground vs background)
#     cm = confusion_matrix(all_gts, all_preds, labels=[0, 1])
#     print("Confusion Matrix (bg ignored in metrics, shown here for sanity):\n", cm)
#
#     # Precision-Recall Curve
#     precision, recall, _ = precision_recall_curve(all_gts, all_preds)
#     ap = average_precision_score(all_gts, all_preds)
#     plt.figure()
#     plt.plot(recall, precision, label=f"AP={ap:.3f}")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("PR Curve (Foreground only)")
#     plt.legend()
#     plt.show()
#
#     # IoU Histogram
#     plt.figure()
#     plt.hist([x for x in per_image_iou if not np.isnan(x)], bins=20, range=(0,1))
#     plt.xlabel("IoU (per image, foreground only)")
#     plt.ylabel("Count")
#     plt.title("Per-image IoU distribution")
#     plt.show()


# def validate(model, val_loader, device, threshold=0.5):
#     model.eval()
#     all_preds = []
#     all_gts = []
#     per_image_ious = []
#
#     with torch.no_grad():
#         for images, masks in val_loader:
#             images = images.to(device)
#             masks = masks.to(device)
#
#             outputs = model(images)
#             probs = torch.sigmoid(outputs).squeeze(1)  # (B, H, W)
#             preds = (probs > threshold).long()
#
#             for pred, gt in zip(preds, masks):
#                 pred_np = pred.cpu().numpy().astype(np.uint8).flatten()
#                 gt_np = gt.cpu().numpy().astype(np.uint8).flatten()
#
#                 all_preds.extend(pred_np.tolist())
#                 all_gts.extend(gt_np.tolist())
#
#                 intersection = np.logical_and(pred_np == 1, gt_np == 1).sum()
#                 union = np.logical_or(pred_np == 1, gt_np == 1).sum()
#                 iou = intersection / union if union > 0 else 0.0
#                 per_image_ious.append(iou)
#    return np.array(all_preds), np.array(all_gts), per_image_ious


def plot_pr_curve(save_path, y_true, y_pred_scores):
    prec, rec, _ = precision_recall_curve(y_true, y_pred_scores)
    plt.figure()
    plt.plot(rec, prec, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Foreground only)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion(save_path, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BG", "FG"])
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_iou_hist(save_path, per_image_ious):
    plt.figure()
    plt.hist(per_image_ious, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.title("Per-image IoU distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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


def create_train_val_test_split(image_dir, mask_dir, transform, random_seed=42):
    """
    Create train/validation/test split by moving files from train to val/test.
    Only original (non-augmented) files are moved to val/test.
    
    Args:
        image_dir: Path to image directory (train folder)
        mask_dir: Path to mask directory (train folder)
        transform: Image transforms
        random_seed: Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset, test_dataset: PyTorch dataset subsets
    """   
    parent_dir = os.path.dirname(image_dir)
    val_image_dir = os.path.join(parent_dir, 'val')
    test_image_dir = os.path.join(parent_dir, 'test')
    val_mask_dir = os.path.join(os.path.dirname(mask_dir), 'val')
    test_mask_dir = os.path.join(os.path.dirname(mask_dir), 'test')
    
    val_has_files = (os.path.exists(val_image_dir) and
                     os.listdir(val_image_dir) and
                     os.path.exists(val_mask_dir) and
                     os.listdir(val_mask_dir))
    test_has_files = (os.path.exists(test_image_dir) and
                      os.listdir(test_image_dir) and
                      os.path.exists(test_mask_dir) and
                      os.listdir(test_mask_dir))
    
    if val_has_files or test_has_files:
        logging.info("Using existing val and test files instead of splitting")

        train_dataset = Loader(image_dir, mask_dir, transform)
        
        if val_has_files:
            val_dataset = Loader(val_image_dir, val_mask_dir, transform)
        else:
            val_dataset = None
            
        if test_has_files:
            test_dataset = Loader(test_image_dir, test_mask_dir, transform)
        else:
            test_dataset = None
            
        return train_dataset, val_dataset, test_dataset
    
    train_image_files = [f for f in os.listdir(image_dir)
                         if '_aug_' not in f and
                         (f.endswith(settings.GEOGRAPHIC_ACCEPT_EXTENSION) or
                          f.endswith(settings.NON_GEOGRAPHIC_ACCEPT_EXTENSION))]
    train_mask_files = [f for f in os.listdir(mask_dir)
                        if '_aug_' not in f and
                        (f.endswith(settings.GEOGRAPHIC_ACCEPT_EXTENSION) or
                         f.endswith(
                             settings.NON_GEOGRAPHIC_ACCEPT_EXTENSION))]
    
    matched_files = []
    for img_file in train_image_files:
        base_name = os.path.splitext(img_file)[0]
        mask_file = None
        for mask in train_mask_files:
            if os.path.splitext(mask)[0] == base_name:
                mask_file = mask
                break
        
        if mask_file:
            matched_files.append((img_file, mask_file))
    
    if not matched_files:
        logging.error("No matching image-mask pairs found in train folder")
        return None, None, None
    
    total_files = len(matched_files)
    val_size = int(settings.VALIDATION_SPLIT * total_files)
    test_size = int(settings.TEST_SPLIT * total_files)
    train_size = total_files - val_size - test_size
    
    random.seed(random_seed)
    random.shuffle(matched_files)
    
    val_files = matched_files[train_size:train_size + val_size]
    test_files = matched_files[train_size + val_size:]
    
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)
    
    for img_file, mask_file in val_files:
        src_img = os.path.join(image_dir, img_file)
        src_mask = os.path.join(mask_dir, mask_file)
        dst_img = os.path.join(val_image_dir, img_file)
        dst_mask = os.path.join(val_mask_dir, mask_file)
        
        shutil.move(src_img, dst_img)
        shutil.move(src_mask, dst_mask)
        logging.info("Moved %s and %s to val folder", img_file, mask_file)
    
    for img_file, mask_file in test_files:
        src_img = os.path.join(image_dir, img_file)
        src_mask = os.path.join(mask_dir, mask_file)
        dst_img = os.path.join(test_image_dir, img_file)
        dst_mask = os.path.join(test_mask_dir, mask_file)
        
        shutil.move(src_img, dst_img)
        shutil.move(src_mask, dst_mask)
        logging.info("Moved %s and %s to test folder", img_file, mask_file)
    
    logging.info("Split completed: %d train, %d val, %d test files",
                 train_size, val_size, test_size)
    
    train_dataset = Loader(image_dir, mask_dir, transform)
    val_dataset = Loader(val_image_dir, val_mask_dir, transform)
    test_dataset = Loader(test_image_dir, test_mask_dir, transform)
    
    return train_dataset, val_dataset, test_dataset


def initialize(load_param, augment_data, is_training, is_validating, is_predicting):
    """

    :param load_param:
    :param augment_data:
    :param is_training:
    :param is_validating:
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

    model_name = settings.MODEL_NAME
    timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M")
    model_path = os.path.join(load_param['save_model_dir'], "deeplabv3-" + model_name + "-" + timestamp + ".pth")
    checkpoint_path = os.path.join(load_param['output_checkpoints'], "deeplabv3-" + model_name + "-" + timestamp + ".pth")
    plot_filepath = os.path.join(load_param['save_plot_dir'], "deeplabv3-" + model_name + "-" + timestamp + ".png")

    if model_name == 'resnet50':
        model = models.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    elif model_name == 'mobilenet':
        model = models.deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
    else:
        logging.error("No model specified. Please set 'model' in settings.py.")
        sys.exit(1)

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
        
        train_dataset, val_dataset, test_dataset = create_train_val_test_split(
            image_dir, mask_dir, transform, random_seed=42
        )

        if val_dataset is None:
            logging.error("Cannot proceed with training - no validation dataset available")
            return

        train_loader = DataLoader(train_dataset, batch_size=load_param['batch_size_training'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=load_param['batch_size_training'], shuffle=False)

        train_iou_history, train_acc_history, train_prec_history, train_rec_history, train_f1_history = [], [], [], [], []
        val_iou_history, val_acc_history, val_prec_history, val_rec_history, val_f1_history = [], [], [], [], []

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=load_param['learning_rate'])

        num_epochs = load_param['epochs']
        best_val_iou = 0.0
        patience = load_param['patience']
        patience_counter = 0
        
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

                iou, acc, prec, rec, f1, preds, gts = compute_metrics(outputs, masks)
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

                    iou, acc, prec, rec, f1, preds, gts = compute_metrics(outputs, masks)
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

            # Early stopping and best model saving
            if val_avg_iou > best_val_iou:
                best_val_iou = val_avg_iou
                patience_counter = 0
                # Save best model
                best_model_path = checkpoint_path.replace('.pth', '_best.pth')
                torch.save(model.state_dict(), best_model_path)
                logging.info(f">>>> New best model saved with IoU: {best_val_iou:.4f}")
            else:
                patience_counter += 1
                logging.info(f">>>> No improvement. Patience: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                logging.info(f">>>> Early stopping triggered after {epoch + 1} epochs")
                break

        logging.info(">>>>>> Training completed. Saving final model in {}...".format(checkpoint_path))
        torch.save(model.state_dict(), checkpoint_path)

        if settings.PLOT_TRAINING:
            plot_training_history(plot_filepath, train_iou_history, val_iou_history, train_acc_history, val_acc_history,
                                  train_prec_history, val_prec_history, train_rec_history, val_rec_history,
                                  train_f1_history, val_f1_history)

    if eval(is_validating):
        logging.info(">> Running validation only...")
        
        val_dataset = create_train_val_test_split(image_dir, mask_dir, transform, random_seed=42)[1]

        if val_dataset is None:
            logging.error("Cannot proceed with validation - no validation dataset available")
            return

        val_loader = DataLoader(val_dataset, batch_size=load_param['batch_size_training'], shuffle=False)

        if load_param['pretrained_weights'] != '':
            model_filename = os.path.join(load_param['output_checkpoints'], load_param['pretrained_weights'])
            model.load_state_dict(torch.load(model_filename, map_location=device), strict=False)
        model.eval()

        all_preds, all_gts, per_image_ious = [], [], []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)["out"]

                iou, acc, prec, rec, f1, preds, gts = compute_metrics(outputs, masks)

                all_preds.extend(preds.tolist())
                all_gts.extend(gts.tolist())
                per_image_ious.append(iou)

        all_preds_np = np.array(all_preds)
        all_gts_np = np.array(all_gts)
        tp = np.sum((all_preds_np == 1) & (all_gts_np == 1))
        fp = np.sum((all_preds_np == 1) & (all_gts_np == 0))
        fn = np.sum((all_preds_np == 0) & (all_gts_np == 1))
        tn = np.sum((all_preds_np == 0) & (all_gts_np == 0))
        iou = tp / (tp + fp + fn + 1e-8)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        logging.info(f">>>> Validation metrics: IoU: {iou:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        metrics_path = os.path.join(load_param['save_plot_dir'], f"deeplabv3-metrics-{settings.MODEL_NAME}-{timestamp}.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"IoU: {iou:.4f}, ")
            f.write(f"Accuracy: {acc:.4f}, ")
            f.write(f"Precision: {prec:.4f}, ")
            f.write(f"Recall: {rec:.4f}, ")
            f.write(f"F1: {f1:.4f}\n")

        if len(np.unique(all_gts_np)) > 1:
            plot_confusion(os.path.join(load_param['save_plot_dir'], f"deeplabv3-confusion_matrix-{settings.MODEL_NAME}-{timestamp}.png"), all_gts_np, all_preds_np)
            plot_pr_curve(os.path.join(load_param['save_plot_dir'], f"deeplabv3-pr_curve-{settings.MODEL_NAME}-{timestamp}.png"), all_gts_np, all_preds_np)
            plot_iou_hist(os.path.join(load_param['save_plot_dir'], f"deeplabv3-iou_hist-{settings.MODEL_NAME}-{timestamp}.png"), per_image_ious)
        else:
            logging.info(">>>> Plots skipped (only one class present in ground truth).")

    if eval(is_predicting):
        logging.info(">> Performing prediction...")

        test_dataset = Loader(load_param['image_prediction_folder'], None, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=load_param['batch_size_prediction'], shuffle=False, collate_fn=collate_fn_predict)

        torch.cuda.empty_cache()
        gc.collect()

        if load_param['pretrained_weights'] != '':
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
