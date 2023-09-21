import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm

from mandrillage.utils import save, load
from mandrillage.dataset import MandrillDualClassificationDataset
from mandrillage.models import FeatureClassificationModel
from mandrillage.pipelines.basic_regression import BasicRegressionPipeline

import logging

log = logging.getLogger(__name__)


class RegressionRankingPipeline(BasicRegressionPipeline):
    def __init__(self):
        super(RegressionRankingPipeline, self).__init__()

    def create_classification_dataset(self, indices, image_dataset):
        dataset = MandrillDualClassificationDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=False,
            max_days=self.max_days,
            same_age_gap=self.same_age_gap,
            n_classes=self.n_classes,
            individuals_ids=indices,
        )
        dataset.set_images(image_dataset.images)
        return dataset

    def init_datamodule(self):
        super().init_datamodule()

        self.ranking_train_dataset = self.create_classification_dataset(
            self.train_indices, self.train_dataset
        )
        self.ranking_val_dataset = self.create_classification_dataset(
            self.val_indices, self.val_dataset
        )

        self.ranking_train_loader = self.make_dataloader(
            self.ranking_train_dataset, shuffle=True
        )
        self.ranking_val_loader = self.make_dataloader(self.ranking_val_dataset)

    def init_logging(self):
        pass

    def init_model(self):
        super().init_model()
        self.ranking_model = FeatureClassificationModel(
            self.backbone,
            input_dim=self.backbone.output_dim,
            n_classes=self.n_classes,
            n_lin=self.classification_stages,
            lin_start=self.classification_lin_start,
        )
        self.ranking_model = self.ranking_model.to(self.device)

    def init_losses(self):
        super().init_losses()
        self.ranking_criterion = nn.CrossEntropyLoss()

    def init_optimizers(self):
        super().init_optimizers()
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.ranking_model.parameters()),
            lr=self.learning_rate,
        )

    def init_callbacks(self):
        pass

    def init_loggers(self):
        pass

    def train_step(self, loader, model, criterion, device):
        x, y = next(iter(loader))
        x, y = self.xy_to_device(x, y, device)

        # Forward pass
        y_hat = model(x)
        loss = criterion(y_hat, y)

        return loss, self.get_size(x)

    def train(self):
        steps = len(self.train_loader)

        if self.resume:
            self.model = load(
                self.model,
                f"regression_{self.train_index}",
                exp_name=self.name,
                output_dir=self.output_dir,
            )
            self.ranking_model = load(
                self.ranking_model,
                f"ranking_{self.train_index}",
                exp_name=self.name,
                output_dir=self.output_dir,
            )

        # Training loop
        best_val = np.inf
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.model.train()  # Set the model to train mode
            self.ranking_model.train()
            train_regression_loss = 0.0
            train_ranking_loss = 0.0
            train_sim_loss = 0.0

            for i in tqdm(range(steps), leave=True):
                self.optimizer.zero_grad()
                reg_loss, reg_size = self.train_step(
                    self.train_loader,
                    self.model,
                    self.criterion,
                    self.device,
                )
                ranking_loss, ranking_size = self.train_step(
                    self.ranking_train_loader,
                    self.ranking_model,
                    self.ranking_criterion,
                    self.device,
                )

                ### SIMILARITY LOSS
                x1, x2 = next(
                    iter(
                        self.train_similarity_loader,
                    )
                )
                x1, x2 = self.xy_to_device(x1, x2, self.device)
                y1, y2 = self.model(x1), self.model(x2)
                sim_loss = self.criterion(y1, y2)

                # Backward pass and optimization
                loss = (
                    0.0 * reg_loss
                    + self.ranking_alpha * ranking_loss
                    + self.sim_alpha * sim_loss
                )
                loss.backward()
                self.optimizer.step()

                train_regression_loss += reg_loss.item() * reg_size
                train_ranking_loss += ranking_loss.item() * ranking_size
                train_sim_loss += sim_loss.item() * self.get_size(x1)

                n_samples = self.batch_size * (i + 1)
                pbar.set_description(
                    f"Train Regression Loss: {(train_regression_loss/n_samples):.5f} - "
                    f"Train Ranking Loss: {(train_ranking_loss/n_samples):.5f} - "
                    f"Train Sim Loss: {(train_sim_loss/n_samples):.5f}"
                )

            train_regression_loss /= len(self.train_dataset)
            train_ranking_loss /= len(self.train_dataset)

            # self.criterion.display_stored_values("train_margin")

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            self.ranking_model.eval()
            val_regression_loss = 0.0
            val_ranking_loss = 0.0
            n_repeat = 1
            val_losses = {}

            with torch.no_grad():
                for val_name, val_fnct in self.val_criterions.items():
                    val_losses[val_name] = self.val_loss(
                        self.val_loader, self.model, val_fnct, self.device
                    )
                    val_losses[val_name] /= len(self.val_dataset)

                val_ranking_loss = self.val_loss(
                    self.ranking_val_loader,
                    self.ranking_model,
                    self.ranking_criterion,
                    self.device,
                    repeat=n_repeat,
                )
                val_ranking_loss /= len(self.val_dataset)

            val_loss = val_losses[self.watch_val_loss]

            if val_loss < best_val:
                log.info(
                    f"Val regression loss improved from {best_val:.4f} to {val_loss:.4f}"
                )
                best_val = val_loss
                save(
                    self.model,
                    f"regression_{self.train_index}",
                    exp_name=self.name,
                    output_dir=self.output_dir,
                )
                save(
                    self.ranking_model,
                    f"ranking_{self.train_index}",
                    exp_name=self.name,
                    output_dir=self.output_dir,
                )
            else:
                log.info(f"Val regression loss did not improved from {best_val:.4f}")

            # Display stored values
            # self.val_criterions["marginloss"].display_stored_values("val_margin")

            # Print training and validation metrics
            val_str = " - ".join(
                [f" val_{name}: {value:.5f}" for name, value in val_losses.items()]
            )
            log.info(
                f"Epoch [{epoch+1}/{self.epochs}] - "
                f"Train Regression Loss: {train_regression_loss:.5f} - "
                f"Train Ranking Loss: {train_ranking_loss:.5f} - "
                f"{val_str} - "
                f"Val Ranking Loss: {val_ranking_loss:.5f}"
            )

    def init_parameters(self):
        super().init_parameters()

        self.classification_lin_start = self.config.model.classification_lin_start
        self.classification_stages = self.config.model.classification_stages
        self.ranking_learning_rate = self.config.training.ranking_learning_rate
        self.n_classes = self.config.dataset.n_classes
        self.same_age_gap = self.config.dataset.same_age_gap
        self.ranking_alpha = self.config.training.ranking_alpha
