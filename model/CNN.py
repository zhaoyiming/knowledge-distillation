import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# Network LeNet
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fcbn1(self.fc1(x)))
        x = self.fc2(x)
        return x


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


# PyTorch经典实现hinton
def loss_sf_kd(outputs, teacher_outputs, params):
    T = params.temperature
    KD_loss = F.kl_div(F.log_softmax(outputs / T, dim=1),
                       F.softmax(teacher_outputs / T, dim=1), reduction="batchmean") * T * T

    return KD_loss


# 原生实现hinton
def loss_sf_kd_sp(outputs, teacher_outputs, params):
    T = params.temperature
    soft_target = F.softmax(teacher_outputs / T, dim=1)
    soft = F.log_softmax(outputs / T, dim=1)
    loss = -torch.mean(torch.sum(soft_target * soft, dim=1)) * T * T
    return loss


# 原生实现soft+hard
def loss_fn_kd_sp(outputs, labels, teacher_outputs, params):
    alpha = params.alpha
    T = params.temperature
    soft_target = F.softmax(teacher_outputs / T, dim=1)
    soft = F.log_softmax(outputs / T, dim=1)

    soft_loss = -torch.mean(torch.sum(soft_target * soft, dim=1))
    hard_loss = F.cross_entropy(outputs, labels)

    loss = soft_loss * T * T * alpha + hard_loss * (1. - alpha)

    return loss


# PyTorch经典实现soft+hard
def loss_fn_kd(outputs, labels, teacher_outputs, params):
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
}
