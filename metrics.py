import torch.nn.functional as F


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE


def MSE(scores, targets):
    MSE = F.mse_loss(scores, targets)
    MSE = MSE.detach().item()
    return MSE
