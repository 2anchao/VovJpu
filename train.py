#coding=utf-8
'''
Copyright (c) 2019, Shining 3D Tech Co., Ltd.
All rights reserved.

Function: whole process of training the segmentation network.

Version: 20190612v1
Author: Jiachen Wu
Revision: the hyper-parameters are obtained from the config.py file.

Version: 20190607v1
Author: Jiachen Wu
Revision: transfer code from Keras to Pytorch.
'''

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from config import config
from vovnet import JPU
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataprocess import Mydataset
from utils.metrics import IoUMetric


max_score = 0   #
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = config.visible_devices

def val(model, device, val_loader, loss, optimizer, metrics, epoch, timestamp):
    global max_score
    model.eval()
    test_loss = 0
    correct = 0
    test_miou = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            y = y.long()
            test_loss += loss(y_hat, y).item()  # sum up batch loss
            test_miou += metrics(y_hat, y)

    test_miou /= len(val_loader)
    test_loss /= len(val_loader)
    writer.add_scalar('Val/Loss', test_loss, epoch)
    writer.add_scalar('Val/Miou', test_miou, epoch)

    print('\nTest set: Average loss: {:.4f}, Miou : {:.4f})\n'.format(
        test_loss, test_miou))
    if max_score < test_miou:
        max_score = test_miou
        os.makedirs('tmp/{}'.format(timestamp), exist_ok=True)
        torch.save(model, 'tmp/{}/{:.4f}_model.path'.format(timestamp, max_score))
    return test_miou


def train(model, device, train_loader, epoch, optimizer, loss, metrics):
    total_trainloss = 0
    total_trainmiou = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):

        x, y = data

        x = x.float().cuda(async=True)
        y = y.cuda(async=True)

        x_var = torch.autograd.Variable(x)
        y_var = torch.autograd.Variable(y)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        try:
            y_hat = model(x_var)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                for i in range(batch_size):
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.transpose(x[i].cpu(), (1, 2, 0)))
                    plt.subplot(1, 2, 2)
                    plt.imshow(y[i].cpu())
                    plt.show()
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception

        train_miou = metrics(y_hat, y.long())
        L = loss(y_hat, y.long())
        L.backward()
        optimizer.step()
        total_trainloss += float(L)
        total_trainmiou += float(train_miou)
        print("batch{}: train_miou:{:.4f} loss:{:.4f}".format(batch_idx, train_miou, L))
        if batch_idx % 10 == 0:
            niter = epoch * len(train_loder) + batch_idx
            writer.add_scalar('Train/Loss', L, niter)
            writer.add_scalar('Train/Miou', train_miou, niter)

    total_trainloss /= len(train_loder)
    total_trainmiou /= len(train_loder)
    print('Train Epoch: {}\t Loss: {:.6f}, Miou: {:.4f}'.format(epoch, total_trainloss, total_trainmiou))



'''
Function: main function
Parameter: None
Output: None
'''
if __name__ == '__main__':

    DEVICE     = 'cuda'
    ACTIVATION = 'softmax'
    timestamp  = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    writer     = SummaryWriter('log/{}'.format(timestamp))

    nb_classes = config.nb_classes
    class_ID   = config.class_ID
    batch_size = config.batch_size
    model_name = config.model_name
    choice_head = config.choice_head

    # determine the dataset
    x_train_dir = config.x_train_dir
    y_train_dir = config.y_train_dir
    x_valid_dir = config.x_valid_dir
    y_valid_dir = config.y_valid_dir

    # pre-process: load data and data normalization
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.533162, 0.378983, 0.323320], [0.067181, 0.052105, 0.042050]) #R_var is 0.061113, G_var is 0.048637, B_var is 0.041166
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.532356, 0.378819, 0.322969], [0.066659,0.051862, 0.041796])  #R_var is 0.061526, G_var is 0.049087, B_var is 0.041330
    ])

    train_dataset = Mydataset(images_dir = x_train_dir, masks_dir = y_train_dir, nb_classes = nb_classes, classes = class_ID,
                              transform = train_transform)
    valid_dataset = Mydataset(images_dir = x_valid_dir, masks_dir = y_valid_dir, nb_classes = nb_classes, classes = class_ID,
                              transform = valid_transform)

    train_loder = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    valid_loder = DataLoader(valid_dataset, batch_size = 1, shuffle = False, num_workers = 4)


    model = JPU()#the choice_head can choice "design" or "build_aspp_decoder"

    criterion = nn.CrossEntropyLoss()
    metrics = IoUMetric(eps=1., activation="softmax2d")

    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.init_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                           eps=1e-08)


    model.cuda()

    for epoch in range(0, config.epoch):        # each epoch

        train(model=model, device=DEVICE, train_loader=train_loder, epoch=epoch, optimizer=optimizer, loss=criterion,
              metrics=metrics)

        test_miou = val(model=model, device=DEVICE, val_loader=valid_loder, loss=criterion, optimizer=optimizer,
                        metrics=metrics, epoch=epoch, timestamp=timestamp)

        scheduler.step(test_miou)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        print("current lr: {}".format(optimizer.param_groups[0]['lr']))

    writer.close()
