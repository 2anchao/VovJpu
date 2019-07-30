'''
Copyright (c) 2019, Shining 3D Tech Co., Ltd.
All rights reserved.

Function: configure the hyper-parameters.

Version: 20190613v1
Author: Yan Tian
Revision: add comments of the variables.

Version: 20190612v1
Author: Jiachen Wu
Revision: the hyper-parameters are obtained from the config.py file.
'''

import datetime

class DefaultConfigs(object):

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    #1.string parameters
    x_train_dir = r'/home/star/ac/large_data/train/Images'
    y_train_dir = r'/home/star/ac/large_data/train/masks'
    x_valid_dir = r'/home/star/ac/large_data/valid/Images'
    y_valid_dir = r'/home/star/ac/large_data/valid/masks'

    pre_weights = r'pretrain_model/mobilenet_v2-6a65762b.pth'
    model_save_path = r'tmp/{}/'.format(timestamp)
    logs = '/logs'
    visible_devices = '0'   # modify according to your hardware

    #2. numeric parameters
    epoch = 120
    per_batch_size = 8  # modify according to your hardware, GTX 1070 uses 6.
    nb_gpus = 1     # GPU cards number, modify according to your hardware.
    batch_size = per_batch_size * nb_gpus
    img_height = 480
    img_width = 640
    nb_classes = 3   #(denatl, gum, background)
    class_ID = [0, 1, 2] #(denatl-0, gum-1, background-2)
    init_lr = 1e-3
    lr_decay = 1e-4
    weight_decay = 5e-4
    model_name = "mobilenet"#'xception resnet101 mobilenet
    choice_head = "design"#"design","build_aspp_decoder"


config = DefaultConfigs()
