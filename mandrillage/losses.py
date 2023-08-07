import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussLoss(nn.Module):
    def __init__(self):
        super(GaussLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """Negative log likelihood of y_true, with the likelihood defined by a normal distribution."""

        means = y_pred[:, 0]
        # We predict the log of the standard deviation, so exponentiate the prediction here
        stds = torch.exp(y_pred[:, 1])
        variances = stds * stds

        log_p = -torch.log(torch.sqrt(2 * math.pi * variances)) - (y_true - means) * (
            y_true - means
        ) / (2 * variances)

        return -log_p


class BinaryRangeMetric(nn.Module):
    def __init__(self):
        super(BinaryRangeMetric, self).__init__()
        self.epsilon = 1 / 365

    def forward(self, y_pred, y):
        y_pred_left = y_pred[..., 0]
        y_pred_length = y_pred[..., 1]
        y_pred_right = y_pred_left + y_pred_length
        y_pred_left = y_pred_left * (y_pred_left > self.epsilon)
        bool_tensor = torch.logical_and(y_pred_left <= y, y <= y_pred_right)
        return torch.mean(bool_tensor.long(), dtype=torch.float)


class RangeLoss(nn.Module):
    """First the interval must be true (i.e. y_lower <= y <= y_upper)
    Then, we try to minimize y_upper - y_lower and center it so abs(y_upper-y_lower - y) == 0


    """

    def __init__(self, alpha=0.1, beta=0.0, epsilon=1e-3):
        super(RangeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # Equivalent to 0.5 day
        self.epsilon = 1 / (365 * 2)

    def forward(self, y_pred, y):
        y_pred_left = y_pred[..., 0]
        y_pred_length = y_pred[..., 1]

        y_pred_right = y_pred_left + y_pred_length

        # Threshold to obtain the zero value otherwise the interval where y=0 is not feasible
        y_pred_left = y_pred_left * (y_pred_left > self.epsilon)

        # print("Y shape: ", y.shape)
        # print("Y_pred shape : ", y_pred.shape)

        null_tensor = torch.tensor(0.0).to(y.device)

        # Minimize the predicted range
        minimize_range = y_pred_length

        # Make sure left is below or equal to y
        left_coherance = torch.max(null_tensor, y_pred_left - y)

        # Make sure right is above of equal to y
        right_coherance = torch.max(null_tensor, y - y_pred_right)

        center_range = (y_pred_right + y_pred_left) / 2
        distance_to_center_range = abs(center_range - y)

        loss = (
            self.alpha * minimize_range
            # + self.beta * distance_to_center_range
            + left_coherance
            + right_coherance
        )
        # left_coherance + right_coherance
        return torch.mean(loss)


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
