import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEVariance(nn.Module):
    def __init__(self):
        super(MSEVariance, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        sigma = torch.mean(torch.var(y_pred))
        mse_value = self.loss(y_pred, y_true)
        return sigma + mse_value


class FeatureSimilarityLoss(nn.Module):
    def __init__(self, eps=1e-9):
        super(FeatureSimilarityLoss, self).__init__()

    def forward(self, x1, x2):
        # x1 = F.normalize(x1, p=2, dim=1)
        # x2 = F.normalize(x2, p=2, dim=1)
        distance = F.pairwise_distance(x1, x2, keepdim=True)
        loss = torch.mean(distance)
        return loss


class AdaptiveMarginLoss(nn.Module):
    def __init__(self, min_days_error, max_days_error, max_days):
        super(AdaptiveMarginLoss, self).__init__()
        # Min age is 0 and max age is 1

        self.min_error = min_days_error / max_days
        self.max_error = max_days_error / max_days
        self.error_ratio = self.max_error / self.min_error

        self.b = self.min_error
        self.a = self.max_error - self.b

        self.loss = nn.L1Loss(reduction="none")
        # self.store_values = False
        self.stored_values = []

        self.weight_b = self.error_ratio
        self.weight_a = 1 - self.weight_b

    def toggle_store_values(self):
        self.store_values = not self.store_values
        if not self.store_values:
            self.stored_values = []

    def to_npy(self, tensor):
        return tensor.detach().cpu().numpy()

    def display_stored_values(self, path):
        import matplotlib.pyplot as plt
        import os

        # Create a figure with margin function
        fig, axs = plt.subplots(3, 1, figsize=(16, 10))
        axs[0].plot([0, 1], [self.min_error, self.max_error])
        axs[1].plot([0, 1], [self.get_weight(0), self.get_weight(1)])
        axs[2].plot([0, 1], [0, 1])

        # For each batch stored
        for errors, weights, preds, ys_true in self.stored_values:
            axs[0].scatter(self.to_npy(ys_true), self.to_npy(errors))
            axs[1].scatter(self.to_npy(ys_true), self.to_npy(weights))
            axs[2].scatter(self.to_npy(ys_true), self.to_npy(preds))
            # for i in range(errors.shape[0]):
            #     error, weight, pred, y_true = (
            #         errors[i].detach().cpu().numpy(),
            #         weights[i].detach().cpu().numpy(),
            #         preds[i].detach().cpu().numpy(),
            #         ys_true[i].detach().cpu().numpy(),
            #     )
            #     # Plot for the given age y_true, the distance to the margin (the error)
            #     axs[0].scatter(y_true, error)
            #     axs[1].scatter(y_true, weight)
            #     axs[2].scatter(y_true, pred)
        axs[0].set_xlim([0, 1])
        axs[0].set_ylim([0, 1])
        axs[1].set_xlim([0, 1])
        axs[1].set_ylim([0, max(self.get_weight(1), 1)])
        axs[2].set_xlim([0, 1])
        axs[2].set_ylim([0, 1])

        plt.savefig(f"{path}.png")
        plt.close()
        self.stored_values = []

    def get_margin(self, x):
        return self.a * x + self.b

    def get_weight(self, x):
        return self.weight_a * x + self.weight_b

    def forward(self, y_pred, y_true):
        # Compute l1 loss
        error = self.loss(y_pred, y_true)
        # Get the threshold margin for all value
        margin = self.get_margin(y_true)
        # Compute distance to margin
        distance = error - margin

        weight = self.get_weight(y_true)
        # if self.store_values:
        self.stored_values.append((error, error * weight, y_pred, y_true))

        # Only penalize samples that are not within the margin distance
        loss = torch.mean(
            torch.max(distance * weight, torch.tensor(0.0).to(y_pred.device))
        )
        return loss


class ScalerLoss(nn.Module):
    def __init__(self, loss, weight):
        super(ScalerLoss, self).__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, y_pred, y_true):
        return torch.mean(self.loss(y_pred, y_true) * self.weight(y_pred, y_true))


class ExponentialWeighting(nn.Module):
    def __init__(self):
        super(ExponentialWeighting, self).__init__()
        self.scale = torch.nn.Parameter(torch.tensor(3.0))
        self.a = torch.nn.Parameter(torch.tensor(0.5))
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.a * torch.exp(-self.scale * y_true)


class LinearWeighting(nn.Module):
    def __init__(self, min_error, max_error, max_days, error_function):
        super(LinearWeighting, self).__init__()

        # Goal is to have the same error value with min_error at zero days
        # and max_error at max days

        zero_days = torch.tensor(0.0)
        max_days = torch.tensor(float(max_days))
        min_error = torch.tensor(float(min_error / max_days))
        max_error = torch.tensor(float(max_error / max_days))

        # Compute both error
        low_end_error = error_function(min_error, zero_days)
        high_end_error = error_function(max_error, zero_days)

        low_end_error = low_end_error.numpy()
        high_end_error = high_end_error.numpy()

        # This is the factor we need to multiply high_end_error by to obtain low_end_error
        self.error_ratio = low_end_error / high_end_error

        # We must solve the linear equation system y = ax + b
        # 1 = a * 0 + b => b = 1
        self.b = 1
        # error_ratio = a * 1 + b
        # error_ratio - b = a
        self._a = self.error_ratio - 1

    @property
    def a(self):
        return self._a

    def get_weight(self, x):
        weight = self.a * x + self.b
        return weight

    def forward(self, y_pred, y_true):
        return self.get_weight(y_true)


def bmc_loss(pred, target, noise_var, device):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    target = torch.unsqueeze(target, dim=-1)
    pred = torch.unsqueeze(pred, dim=-1)
    logits = -(pred - target.T).pow(2) / (2 * noise_var)  # logit size: [batch, batch]
    loss = F.cross_entropy(
        logits, torch.arange(pred.shape[0]).to(device)
    )  # contrastive-like loss
    loss = (
        loss * (2 * noise_var).detach()
    )  # optional: restore the loss scale, 'detach' when noise is learnable

    return loss


class BMCLoss(nn.Module):
    def __init__(self, init_noise_sigma, device):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
        self.device = device

    def forward(self, pred, target):
        noise_var = self.noise_sigma**2
        return bmc_loss(pred, target, noise_var, self.device)


class GaussLoss(nn.Module):
    def __init__(self):
        super(GaussLoss, self).__init__()

    def forward(self, y_pred, y):
        mu = y_pred[:, 0]  # first output neuron
        log_sig = y_pred[:, 1]  # second output neuron
        sig = torch.exp(log_sig)  # undo the log
        return torch.mean(2 * log_sig + ((y - mu) / sig) ** 2)


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

    def __init__(self, alpha=0.01, beta=0.0, epsilon=1e-3):
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
