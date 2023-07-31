import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from mandrillage.utils import split_dataset, save, load
from mandrillage.dataset import MandrillDualClassificationDataset
from mandrillage.models import FeatureClassificationModel
from mandrillage.pipelines.regression_ranking import RegressionRankingPipeline

import logging

log = logging.getLogger(__name__)


class RegressionRankingCombinedPipeline(RegressionRankingPipeline):
    def __init__(self):
        super(RegressionRankingCombinedPipeline, self).__init__()

    def train_step(self, loader, optimizer, model, criterion, device):
        x, y = next(iter(loader))
        x, y = self.xy_to_device(x, y, device)
        optimizer.zero_grad()

        # Forward pass

        y_hat = model(x)
        loss = criterion(y_hat, y)

        return loss, self.get_size(x)

    def train(self):
        steps = len(self.train_loader)

        if self.resume:
            self.model = load(self.model, "regression", exp_name=self.name)
            self.ranking_model = load(self.ranking_model, "ranking", exp_name=self.name)

        # Training loop
        best_val = np.inf
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.model.train()  # Set the model to train mode
            self.ranking_model.train()
            train_regression_loss = 0.0
            train_ranking_loss = 0.0

            for i in tqdm(range(steps), leave=True):
                reg_loss, reg_size = self.train_step(
                    self.train_loader,
                    self.optimizer,
                    self.model,
                    self.criterion,
                    self.device,
                )
                ranking_loss, ranking_size = self.train_step(
                    self.ranking_train_loader,
                    self.ranking_optimizer,
                    self.ranking_model,
                    self.ranking_criterion,
                    self.device,
                )

                # Backward pass and optimization
                loss = reg_loss + ranking_loss
                loss.backward()
                self.optimizer.step()
                self.ranking_optimizer.step()

                train_regression_loss += reg_loss.item() * reg_size
                train_ranking_loss += ranking_loss.item() * ranking_size

                n_samples = self.batch_size * (i + 1)
                pbar.set_description(
                    f"Train Regression Loss: {(train_regression_loss/n_samples):.5f} - "
                    f"Train Ranking Loss: {(train_ranking_loss/n_samples):.5f}"
                )

            train_regression_loss /= len(self.train_dataset)
            train_ranking_loss /= len(self.train_dataset)

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            self.ranking_model.eval()
            val_regression_loss = 0.0
            val_ranking_loss = 0.0
            n_repeat = 1

            with torch.no_grad():
                val_regression_loss = self.val_loss(
                    self.val_loader, self.model, self.val_criterion, self.device
                )
                val_ranking_loss = self.val_loss(
                    self.ranking_val_loader,
                    self.ranking_model,
                    self.ranking_criterion,
                    self.device,
                    repeat=n_repeat,
                )
            val_regression_loss /= len(self.val_dataset)
            val_ranking_loss /= len(self.val_dataset)

            if val_regression_loss < best_val:
                log.info(
                    f"Val regression loss improved from {best_val:.4f} to {val_regression_loss:.4f}"
                )
                best_val = val_regression_loss
                save(self.model, "regression", exp_name=self.name)
                save(self.ranking_model, "ranking", exp_name=self.name)
            else:
                log.info(f"Val regression loss did not improved from {best_val:.4f}")

            # Print training and validation metrics
            log.info(
                f"Epoch [{epoch+1}/{self.epochs}] - "
                f"Train Regression Loss: {train_regression_loss:.5f} - "
                f"Train Ranking Loss: {train_ranking_loss:.5f} - "
                f"Val Regression Loss: {val_regression_loss:.5f} - "
                f"Val Ranking Loss: {val_ranking_loss:.5f}"
            )

    def init_parameters(self):
        super().init_parameters()

        self.classification_lin_start = self.config.model.classification_lin_start
        self.classification_stages = self.config.model.classification_stages
        self.ranking_learning_rate = self.config.training.ranking_learning_rate
        self.n_classes = self.config.dataset.n_classes
        self.same_age_gap = self.config.dataset.same_age_gap
