import torch.nn as nn
import torch
import numpy as np

class IoUMetric(nn.Module):

    __name__ = 'iou'

    def __init__(self, eps=1e-7, threshold=0.5, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps
        self.threshold = threshold

    def forward(self, y_pr, y_gt):
        return iou(y_pr, y_gt, self.eps, self.threshold, self.activation)


"""
    Source:
        https://github.com/catalyst-team/catalyst/
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
"""
def iou(pr, gt, eps=1e-7, threshold=None, activation='sigmoid'):

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)
    iou_all = 0
    smooth = 1

    pr = torch.argmax(pr, dim=1)
    pr = pr.cpu().numpy()
    gt = gt.cpu().numpy()

    pr = to_categorical(pr, num_classes=3)
  #  pr = np.transpose(pr, (0, 3, 1, 2))
    gt = to_categorical(gt, num_classes=3)
  #  gt = np.transpose(gt, (0, 3, 1, 2))
    # print(pr.shape, gt.shape, 'test')

    nb_classes = 3
    for i in range(0, nb_classes):      # each class
      #  res_true = np.where(np.equal(gt, i), np.ones_like(gt), np.zeros_like(gt))
        res_true = gt[:, :, :, i:i + 1]
        res_pred = pr[:, :, :, i:i + 1]

        res_pred = res_pred.astype(np.float64)
        res_true = res_true.astype(np.float64)

        intersection = np.sum(np.abs(res_true * res_pred), axis=(1, 2, 3))
        union = np.sum(res_true, axis=(1, 2, 3)) + np.sum(res_pred, axis=(1, 2, 3)) - intersection
        iou_all += (np.mean((intersection + smooth) / (union + smooth), axis=0))

    return iou_all / nb_classes


"""
    Converts a class vector (integers) to binary class matrix.

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
def to_categorical(y, num_classes=None, dtype='float32'):

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
