'''
Copyright (c) 2019, Shining 3D Tech Co., Ltd.
All rights reserved.

Function: pre-process.

Version: 20190607v1
Author: Jiachen Wu
Revision: transfer code from Keras to Pytorch.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import random


'''
Function: read data
Parameter: images_dir: image address
           masks_dir: annotation file address
           nb_classes: class number
           classes: class ID
           transform: data augmentation methods
Output: None
'''
class Mydataset(Dataset):

    CLASSES = [0, 1, 2]

    def __len__(self):
        return len(self.ids)

    def __init__(self, images_dir:str, masks_dir:str, nb_classes, classes = None, transform = None):
        super().__init__()
        self.class_values = [self.CLASSES.index(cls) for cls in classes]
        self.nb_classes = nb_classes
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, os.path.splitext(image_id)[0] + '.npy') for image_id in self.ids]
        self.transform = transform

    def __getitem__(self, i):
        # read data
        image = Image.open(self.images_fps[i])

        mask = np.load(self.masks_fps[i])
        mask[mask > self.nb_classes - 1] = 0
        # data aug
        mask=Image.fromarray(mask)
        j=random.choice([0,1,2])
        if j==0:
            image=image.transpose(Image.FLIP_LEFT_RIGHT)
            mask=mask.transpose(Image.FLIP_LEFT_RIGHT)
        elif j==1:
            image=image.transpose(Image.FLIP_TOP_BOTTOM)
            mask=mask.transpose(Image.FLIP_TOP_BOTTOM)
        elif j==2:
            pass
        mask=np.array(mask)

        if self.transform is not None:
            image = self.transform(image)
        return image, mask



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

    train_transform=transforms.Compose([
        #transforms.RandomVerticalFlip(),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.519401, 0.359217, 0.310136], [0.061113, 0.048637, 0.041166]),#R_var is 0.061113, G_var is 0.048637, B_var is 0.041166
    ])
    valid_transform=transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize([0.517446, 0.360147, 0.310427], [0.061526,0.049087, 0.041330])#R_var is 0.061526, G_var is 0.049087, B_var is 0.041330
    ])

    train_dataset = Mydataset(images_dir=x_train_dir, masks_dir=y_train_dir, nb_classes= 3,classes=[0,1,2], transform=train_transform)
    valid_dataset = Mydataset(images_dir=x_valid_dir, masks_dir=y_valid_dir, nb_classes= 3,classes=[0,1,2], transform=valid_transform)
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

        plt.imshow(mask)
        plt.show()
        exit()
