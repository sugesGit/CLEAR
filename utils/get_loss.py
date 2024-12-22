import torch
from torch import nn

def get_loss(y_pred, y_true):
    loss = nn.CrossEntropyLoss()
    return loss(y_pred, y_true)

def get_loss_weighted(y_pred, y_true, weights):
    y_true = y_true.float()
    weights = weights.float()
    loss_fn = nn.BCELoss(weight=weights)
    return loss_fn(y_pred, y_true)
