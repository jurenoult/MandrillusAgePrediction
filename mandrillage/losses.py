import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossAdaptiveMargin(nn.Module):
    def __init__(self, eps=1e-9):
        super(TripletLossAdaptiveMargin, self).__init__()

    def forward(self, anchor, positive, negative, adaptive_margin):
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        distance_positive = F.pairwise_distance(anchor, positive, keepdim=True)
        distance_negative = F.pairwise_distance(anchor, negative, keepdim=True)

        distance = distance_positive - distance_negative + adaptive_margin

        loss = torch.mean(torch.max(distance, torch.tensor(0.0).to(anchor.device)))
        return loss


class ExponentialDecayLoss(nn.Module):
    def __init__(self):
        super(ExponentialDecayLoss, self).__init__()
        self.scale = 3
        self.a = 0.5
        self.mse = nn.MSELoss()

    def forward(self, predicted, truth):
        t = truth
        p = predicted
        scale = self.a * torch.exp(-self.scale * t)
        error = self.mse(predicted, truth)

        relative_error = scale * error
        return torch.mean(relative_error)


class LinearDecayLoss(nn.Module):
    def __init__(self):
        super(LinearDecayLoss, self).__init__()
        self.slope = 4
        self.intercept = 1
        self.mse = nn.MSELoss()

    def forward(self, predicted, truth):
        scale = (1.0 - truth) * self.slope + self.intercept
        error = self.mse(predicted, truth)
        scaled_error = scale * error

        return torch.mean(scaled_error)
