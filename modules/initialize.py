import os
import sys
import torch
import settings
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models.segmentation as models

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataloader import Loader

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader


def list_entries(directory):
    """
    Image training and validation loader. List the entries in the folder and its subfolders

    :param directory:
    :return input_img_paths: list
    """
    input_img_paths = sorted([os.path.join(root, name)
                              for root, dirs, files in os.walk(directory)
                              for name in files
                              if (name.endswith(settings.NON_GEOGRAPHIC_ACCEPT_EXTENSION)) and not
                              name.startswith(".")])

    print(">>>> Number of samples: {}".format(len(input_img_paths)))
    return input_img_paths


def compute_metrics(preds, masks):
    """Computes IoU and Accuracy."""
    preds = (torch.sigmoid(preds) > 0.5).float()

    intersection = (preds * masks).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + masks.sum(dim=(1, 2)) - intersection

    # Avoid NaN: If union is zero, set IoU to 1 (perfect match) or 0 (no overlap)
    iou = torch.where(union > 0, intersection / union, torch.tensor(1.0)).mean().item()

    accuracy = (preds == masks).float().mean().item()
    return iou, accuracy


def initialize(load_param, augment_data, is_training, is_predicting):
    """

    :param load_param:
    :param augment_data:
    :param is_training:
    :param is_predicting:
    :return:
    """
    transform = T.Compose([
        T.Resize((128, 128)),
        T.CenterCrop(128),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if eval(is_training):
        print(">> Loading input datasets...")
        image_dir = load_param['image_training_folder']
        mask_dir = load_param['annotation_training_folder']

        dataset = Loader(image_dir, mask_dir, transform)

        val_size = int(settings.VALIDATION_SPLIT * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=load_param['batch_size'], shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=load_param['batch_size'], shuffle=False)

        # if eval(augment_data):
        #     logging.info(">> Augmenting entries...")
        #
        #     train_images_paths, train_labels_paths = remove_already_augmented(train_images_paths, train_labels_paths)
        #
        #     augmentor = augment.Augment(img_size, train_images_paths, train_labels_paths)
        #     augmentor.augment()
        #     train_images_paths = augmentor.train_image_paths
        #     train_labels_paths = augmentor.train_labels_paths

        # model = models.deeplabv3_resnet50(pretrained=True)
        model = models.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=load_param['learning_rate'])

        num_epochs = load_param['epochs']
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_iou = 0.0
            total_accuracy = 0.0

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)["out"]
                loss = criterion(outputs.squeeze(1), masks.squeeze(1))

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                iou, acc = compute_metrics(outputs, masks)
                total_iou += iou
                total_accuracy += acc

            avg_loss = running_loss / len(train_loader)
            avg_iou = total_iou / len(train_loader)
            avg_acc = total_accuracy / len(train_loader)
            print(f">>>> Epoch {epoch + 1} | Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Accuracy: {avg_acc:.4f}")

        timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M")
        model_path = os.path.join(load_param['save_model_dir'], "deeplabv3-" + timestamp + ".pth")

        print(">>>>>> Model built. Saving model in {}...".format(model_path))
        torch.save(model.state_dict(), model_path)

    if is_predicting:
        print(">> Performing prediction...")

        model = models.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

        model.load_state_dict(torch.load(os.path.join(load_param['save_model_dir'],
                                                      load_param['pretrained_weights']), weights_only=True))

        test_images_list = list_entries(load_param['image_prediction_folder'])
        images = [transform(Image.open(img_path).convert("RGB")) for img_path in test_images_list]
        batch = torch.stack(images)

        model.to(device)
        batch = batch.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(batch)["out"]

        for img_path, prediction in zip(test_images_list, outputs.squeeze(1)):
            filename = os.path.basename(img_path)

            pred_image = torch.sigmoid(prediction).cpu().numpy() > 0.5

            # orig_size = (settings.ORIGINAL_SIZE, settings.ORIGINAL_SIZE)
            # pred_resized = TF.resize(Image.fromarray((pred_image * 255).astype('uint8')), orig_size)
            # pred_resized = np.array(pred_resized)

            Image.fromarray(pred_image).save(os.path.join(load_param['output_prediction'], filename))
