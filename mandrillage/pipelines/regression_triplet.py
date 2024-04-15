import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm

from mandrillage.utils import save, load
from mandrillage.dataset import MandrillTripleImageDataset
from mandrillage.losses import TripletLossAdaptiveMargin
from mandrillage.pipelines.basic_regression import BasicRegressionPipeline

import logging

log = logging.getLogger(__name__)


class RegressionTripletPipeline(BasicRegressionPipeline):
    def __init__(self):
        super(RegressionTripletPipeline, self).__init__()

    def create_classification_dataset(self, indices, image_dataset):
        dataset = MandrillTripleImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=False,
            max_days=self.max_days,
            individuals_ids=indices,
        )
        dataset.set_images(image_dataset.images)
        return dataset

    def init_datamodule(self):
        super().init_datamodule()

        self.triplet_train_dataset = self.create_classification_dataset(
            self.train_indices, self.train_dataset
        )
        self.triplet_val_dataset = self.create_classification_dataset(
            self.val_indices, self.val_dataset
        )

        self.triplet_train_loader = self.make_dataloader(
            self.triplet_train_dataset, shuffle=True
        )
        self.triplet_val_loader = self.make_dataloader(self.triplet_val_dataset)

    def init_logging(self):
        pass

    def init_model(self):
        super().init_model()

    def init_losses(self):
        super().init_losses()
        print(self.val_criterions)
        self.triplet_criterion = TripletLossAdaptiveMargin()
        self.triplet_val_criterion = TripletLossAdaptiveMargin()

    def init_optimizers(self):
        super().init_optimizers()

    def init_callbacks(self):
        pass

    def init_loggers(self):
        pass

    def triplet_step(self, loader, model, criterion):
        (i1, i2, i3), margin = next(iter(loader))

        # Forward pass
        i1, i2, i3, margin = (
            i1.to(self.device),
            i2.to(self.device),
            i3.to(self.device),
            margin.to(self.device),
        )
        o1, o2, o3 = model(i1), model(i2), model(i3)
        loss = criterion(o1, o2, o3, margin)

        return loss, i1.size(0)

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
            self.model = load(
                self.model, "regression", exp_name=self.name, output_dir=self.output_dir
            )

        # Training loop
        best_val = np.inf
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.model.train()  # Set the model to train mode
            train_regression_loss = 0.0
            train_triplet_loss = 0.0

            for i in tqdm(range(steps), leave=True):
                reg_loss, reg_size = self.train_step(
                    self.train_loader,
                    self.optimizer,
                    self.model,
                    self.criterion,
                    self.device,
                )
                triplet_loss, triplet_size = self.triplet_step(
                    self.triplet_train_loader,
                    self.backbone,
                    self.triplet_criterion,
                )

                # Backward pass and optimization
                loss = 0.000 * reg_loss + self.triplet_alpha * triplet_loss
                loss.backward()
                self.optimizer.step()

                train_regression_loss += reg_loss.item() * reg_size
                train_triplet_loss += triplet_loss.item() * triplet_size

                n_samples = self.batch_size * (i + 1)
                pbar.set_description(
                    f"Train Regression Loss: {(train_regression_loss/n_samples):.5f} - "
                    f"Train Triplet Loss: {(train_triplet_loss/n_samples):.5f}"
                )

            train_regression_loss /= len(self.train_dataset)
            train_triplet_loss /= len(self.train_dataset)

            self.criterion.display_stored_values("train_margin")

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            val_regression_loss = 0.0
            val_triplet_loss = 0.0
            n_repeat = 1

            with torch.no_grad():
                val_regression_loss = self.val_loss(
                    self.val_loader,
                    self.model,
                    self.val_criterions[self.watch_val_loss],
                    self.device,
                )

                for i in range(len(self.triplet_val_loader)):
                    triplet_loss, triplet_size = self.triplet_step(
                        self.triplet_val_loader,
                        self.backbone,
                        self.triplet_val_criterion,
                    )
                    val_triplet_loss += triplet_loss.item() * triplet_size

            self.val_criterions["marginloss"].display_stored_values("val_margin")

            val_regression_loss /= len(self.val_dataset)
            val_triplet_loss /= len(self.val_dataset)

            if val_regression_loss < best_val:
                log.info(
                    f"Val regression loss improved from {best_val:.4f} to {val_regression_loss:.4f}"
                )
                best_val = val_regression_loss
                save(
                    self.model,
                    "regression",
                    exp_name=self.name,
                    output_dir=self.output_dir,
                )
            else:
                log.info(f"Val regression loss did not improved from {best_val:.4f}")

            # Print training and validation metrics
            log.info(
                f"Epoch [{epoch+1}/{self.epochs}] - "
                f"Train Regression Loss: {train_regression_loss:.5f} - "
                f"Train Triplet Loss: {train_triplet_loss:.5f} - "
                f"Val Regression Loss: {(val_regression_loss*self.max_days):.5f} - "
                f"Val Triplet Loss: {val_triplet_loss:.5f}"
            )

    def init_parameters(self):
        super().init_parameters()
        self.triplet_alpha = self.config.training.triplet_alpha
