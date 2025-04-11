import os
import logging
import numpy as np
import settings

from PIL import Image
from imgaug import augmenters as iaa


class Augment:

    def __init__(self, img_size, train_image_paths, train_labels_paths):
        self.img_size = img_size
        self.train_image_paths = train_image_paths
        self.train_labels_paths = train_labels_paths

    def get_augment_seq(self, aug_type):
        """
        Prepare augmentation and return the respective aug_type object

        :param aug_type: string describing the augmentation type
        :return: imgaug object

        Source:
            - https://www.programcreek.com/python/example/115046/imgaug.HooksImages
            - https://www.programcreek.com/python/?code=JohnleeHIT%2FBrats2019%2FBrats2019-master%2Fsrc%2Futils.py#
        """
        if aug_type == 'all':
            seq = iaa.Sequential([
                iaa.Fliplr(1.0),  # horizontally flip 50% of all images
                iaa.Flipud(1.0),  # vertically flip 20% of all images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Affine(rotate=(-45, 45)),
                iaa.Affine(scale=(0.5, 1.3)),
                iaa.Dropout([0.05, 0.2]),
                iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
                iaa.Affine(scale=(1.5, 2.3)),
                iaa.Affine(scale=(0.4, 1.2)),  # rotate by -45 to 45 degrees (affects segmaps)
            ], random_order=True)
        elif aug_type == 'rotation':
            seq = iaa.Sequential([
                iaa.Fliplr(1.0),  # horizontally flip 50% of all images
                iaa.Flipud(1.0),  # vertically flip 20% of all images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
            ], random_order=True)
        elif aug_type == 'noise':
            seq = iaa.Sequential([
                iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
            ], random_order=True)
        elif aug_type == 'blured':
            seq = iaa.Sequential([
                iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
            ], random_order=True)
        elif aug_type == 'resize-1':
            seq = iaa.Sequential([
                iaa.Affine(scale=(1.5, 2.3))
            ], random_order=True)
        elif aug_type == 'resize-2':
            seq = iaa.Sequential([
                iaa.Affine(scale=(0.4, 1.2))
            ], random_order=True)
        elif aug_type == 'color-1':
            seq = iaa.Sequential([
                iaa.MultiplyBrightness((0.5, 1.0)),
            ], random_order=True)
        elif aug_type == 'color-2':
            seq = iaa.Sequential([
                iaa.MultiplyBrightness((1.1, 1.8)),
            ], random_order=True)
        else:
            seq = iaa.Sequential([
                iaa.Affine(scale=0.5),  # rotate by -45 to 45 degrees (affects segmaps)
            ], random_order=True)
        return seq

    def image_aug_filename(self, path, aug_type):
        """
        Setup filename to the augmented image

        :param path: absolute path to original file
        :param aug_type: type of augment effect
        :return: the new augmented image filename
        """
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)
        name, extension = filename.split('.')
        image_aug_filename = os.path.join(dirname, name + "_aug_" + aug_type + "." + extension)

        return image_aug_filename

    def augment(self):
        """
        Get all images entries and apply augmentation according to types variable
        Source:
            - https://www.programcreek.com/python/example/127234/tifffile.imsave
            - https://stackoverflow.com/questions/53776506/how-to-save-an-array-representing-an-image-with-40-band-to-a-tif-file
        """
        train_image_paths_aux = self.train_image_paths.copy()

        for t in settings.AUGMENTATION_TRANSFORMS:
            seq = self.get_augment_seq(t)
            det = seq.to_deterministic()

            train_images = []
            label_images = []
            logging.info(">>>> Augmenting with {} effects...".format(t))
            for j in range(0, len(train_image_paths_aux)):

                x = Image.open(self.train_image_paths[j]).convert("RGB")
                x = np.array(x)
                x = det.augment_image(x)

                if settings.LABEL_TYPE == 'rgb':
                    y = Image.open(self.train_labels_paths[j]).convert("RGB")
                    y = np.array(y)
                else:
                    y = Image.open(self.train_labels_paths[j]).convert("L")
                    y = np.array(y)
                    y = np.expand_dims(y, 2)

                # If x is more than 3-channels image, the augmentation will fail for color-like augments
                if 'color-1' not in t and 'color-2' not in t and 'all' not in t:
                    y = det.augment_image(y)

                image_aug_filename = self.image_aug_filename(self.train_image_paths[j], t)
                label_aug_filename = self.image_aug_filename(self.train_labels_paths[j], t)

                im_x = Image.fromarray(x)
                y = np.squeeze(y, axis=2)
                im_y = Image.fromarray(y)

                im_x.save(image_aug_filename, "TIFF")
                im_y.save(label_aug_filename)

                train_images.append(image_aug_filename)
                label_images.append(label_aug_filename)

            self.train_image_paths.extend(train_images)
            self.train_labels_paths.extend(label_images)

