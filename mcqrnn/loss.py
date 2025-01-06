import torch
import torch.nn as nn


def composite_check_loss(predictions, targets, quantile, alpha):
    tau = quantile

    errors = targets - predictions
    huber = (errors.pow(2) / 2 / alpha) * (errors.abs() <= alpha) + (
        errors.abs() - alpha / 2
    ) * (errors.abs() > alpha)

    check_loss = (
        torch.sum((tau * huber)[errors >= 0])
        + torch.sum(((1 - tau) * huber)[errors < 0])
    ) / huber.size()[0]

    return check_loss
