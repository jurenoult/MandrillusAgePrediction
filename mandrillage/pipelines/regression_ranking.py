import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from mandrillage.utils import split_dataset, save, load
from mandrillage.dataset import MandrillDualClassificationDataset
from mandrillage.models import FeatureClassificationModel
from mandrillage.pipelines.basic_regression import BasicRegressionPipeline

import logging

log = logging.getLogger(__name__)


class RegressionRankingPipeline(BasicRegressionPipeline):
    def __init__(self):
        super(RegressionRankingPipeline, self).__init__()

    def init_datamodule(self):
        super().init_datamodule()
        ranking_dataset = MandrillDualClassificationDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=False,
            max_days=self.max_days,
        )
        ranking_dataset.set_images(self.dataset.images)

        (
            self.ranking_train_loader,
            self.ranking_val_loader,
            self.ranking_train_dataset,
            self.ranking_val_dataset,
        ) = split_dataset(
            ranking_dataset, self.train_ratio, self.batch_size, augment=False
        )

    def init_logging(self):
        pass

    def init_model(self):
        super().init_model()
        self.ranking_model = FeatureClassificationModel(
            self.backbone,
            input_dim=self.backbone.output_dim,
            n_classes=3,
            n_lin=self.classification_stages,
            lin_start=self.classification_lin_start,
        )
        self.ranking_model = self.ranking_model.to(self.device)

    def init_losses(self):
        super().init_model()
        self.ranking_criterion = nn.CrossEntropyLoss()

    def init_optimizers(self):
        super().init_optimizers()
        self.ranking_optimizer = optim.Adam(
            self.ranking_model.parameters(), lr=self.ranking_learning_rate
        )

    def init_callbacks(self):
        pass

    def init_loggers(self):
        pass

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
                train_regression_loss += self.train_step(
                    self.train_loader,
                    self.optimizer,
                    self.model,
                    self.criterion,
                    self.device,
                )
                train_ranking_loss += self.train_step(
                    self.ranking_train_loader,
                    self.ranking_optimizer,
                    self.ranking_model,
                    self.ranking_criterion,
                    self.device,
                )

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

            with torch.no_grad():
                val_regression_loss = self.val_loss(
                    self.val_loader, self.model, self.val_criterion, self.device
                )
                val_ranking_loss = self.val_loss(
                    self.ranking_val_loader,
                    self.ranking_model,
                    self.ranking_criterion,
                    self.device,
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
                f"Val Regression Loss: {val_regression_loss:.5f}"
                f"Val Ranking Loss: {val_ranking_loss:.5f}"
            )

    def init_parameters(self):
        super().init_parameters()

        self.classification_lin_start = self.config.model.classification_lin_start
        self.classification_stages = self.config.model.classification_stages
        self.ranking_learning_rate = self.config.training.ranking_learning_rate
