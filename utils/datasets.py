from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from skimage.io import imread
import imgaug.augmenters as iaa
import imgaug as ia

import cv2
import matplotlib.pyplot as plt
import torch
class dentaldata(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """
    # CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
    #            'tree', 'signsymbol', 'fence', 'car',
    #            'pedestrian', 'bicyclist', 'unlabelled']
    CLASSES = [0,1,2]


    def __init__(
            self,
            images_dir,
            masks_dir,
            nb_classes,
            classes=None,
            augmentation=False,
            preprocessing=True,
            shape = (480, 640)
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0]+'.npy') for image_id in self.ids]
        self.shape = shape
        self.seq = iaa.Sequential([iaa.Fliplr(0.5),
                                   iaa.Affine(translate_percent=(0, 0.1)),
                                   iaa.Affine(rotate=(-15, 15)),
                                   # iaa.Affine(scale=(0.8, 1.2))
                                   ], random_order=True)
        self.seq_det = self.seq.to_deterministic()
        self.classes = classes
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls) for cls in classes]
        self.nb_classes = nb_classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = imread(self.images_fps[i])

        mask = np.load(self.masks_fps[i])
        mask[mask > self.nb_classes-1] = 0
        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')


        # apply augmentations
        if self.augmentation:
            image = self.seq_det.augment_image(image)
            #
            segmap = ia.SegmentationMapOnImage(mask, shape=self.shape, nb_classes=self.nb_classes)

            # label = self.seq.augment_image(label)
            mask = self.seq_det.augment_segmentation_maps([segmap])[0].get_arr_int().astype(np.uint8)

        # apply preprocessing
        if self.preprocessing:
            image = image.astype('float')/255
        #print(image.shape,'test')
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        #mask = to_categorical(mask, num_classes=self.nb_classes)
#        mask = np.transpose(mask,(2,0,1)).astype(np.float32)
        #convert one-hot matrix
        #mask = to_categorical(mask,num_classes=self.nb_classes)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        return image, mask

    def __len__(self):
        return len(self.ids)

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

if __name__ == '__main__':
    x_train_dir = r'C:\Users\fenglian\Desktop\data\5training\imgs'
    y_train_dir = r'C:\Users\fenglian\Desktop\data\5training\masks'
    x_valid_dir = r'C:\Users\fenglian\Desktop\data\5validation\imgs'
    y_valid_dir = r'C:\Users\fenglian\Desktop\data\5validation\masks'

    train_dataset = dentaldata(images_dir=x_train_dir, masks_dir=y_train_dir, nb_classes= 3,classes=[0,1,2], augmentation=True)
    valid_dataset = dentaldata(images_dir=x_valid_dir, masks_dir=y_valid_dir, nb_classes= 3,classes=[0,1,2], augmentation=False)
    train_loder = DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers=4)
    valid_loder = DataLoader(valid_dataset, batch_size = 1, shuffle = False, num_workers=4)
    for data in train_loder:
        #print(data[0][0].numpy().shape)


        img2 = data[0][0].numpy()*255
        img2 = img2.astype('uint8')

        print(img2.shape)

        plt.imshow(np.transpose(img2,(1,2,0)))

        plt.show()
        #one-hot matrix to vector
        mask = data[1][0].numpy()

        print(mask.shape)

        plt.imshow(np.transpose(mask,(1,2,0)))
        plt.show()
        exit()


