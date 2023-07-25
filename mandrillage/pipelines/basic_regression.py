import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from mandrillage.utils import split_dataset, save, load
from mandrillage.dataset import read_dataset, MandrillImageDataset
from mandrillage.models import RegressionModel, VGGFace
from mandrillage.pipeline import Pipeline
from mandrillage.evaluations import standard_regression_evaluation
from torch.utils.data import DataLoader


class BasicRegressionPipeline(Pipeline):
    def __init__(self):
        super(BasicRegressionPipeline, self).__init__()

    def init_datamodule(self):
        self.data = read_dataset(
            self.dataset_metadata_path,
            filter_dob_error=True,
            filter_certainty=self.config.dataset.dob_certain_only,
            max_age=self.max_days,
        )
        self.dataset = MandrillImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=self.in_mem,
            max_days=self.max_days,
        )

        (
            self.train_loader,
            self.val_loader,
            self.train_dataset,
            self.val_dataset,
        ) = split_dataset(
            self.dataset, self.train_ratio, self.batch_size, augment=False
        )

    def init_logging(self):
        pass

    def init_model(self):
        self.backbone = VGGFace()
        self.model = RegressionModel(
            self.backbone,
            input_dim=self.backbone.output_dim,
            lin_start=self.regression_lin_start,
            n_lin=self.regression_stages,
        )
        self.backbone = self.backbone.to(self.device)
        self.model = self.model.to(self.device)

    def init_losses(self):
        # Losses
        self.criterion = nn.MSELoss()
        self.val_criterion = nn.L1Loss()

    def init_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def init_callbacks(self):
        pass

    def init_loggers(self):
        pass

    def train(self):
        steps = len(self.train_loader)

        if self.resume:
            self.model = load(self.model, "regression", exp_name=self.name)

        # Training loop
        best_val = np.inf
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.model.train()  # Set the model to train mode
            train_loss = 0.0

            for i in tqdm(range(steps), leave=True):
                train_loss += self.train_step(
                    self.train_loader,
                    self.optimizer,
                    self.model,
                    self.criterion,
                    self.device,
                )

                n_samples = self.batch_size * (i + 1)
                pbar.set_description(f"Train Loss: {(train_loss/n_samples):.5f}")

            train_loss /= len(self.train_dataset)

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            val_loss = 0.0

            with torch.no_grad():
                val_loss = self.val_loss(
                    self.val_loader, self.model, self.val_criterion, self.device
                )
            val_loss /= len(self.val_dataset)

            if val_loss < best_val:
                print(f"Val loss improved from {best_val:.4f} to {val_loss:.4f}")
                best_val = val_loss
                save(self.model, "regression", exp_name=self.name)
            else:
                print(f"Val loss did not improved from {best_val:.4f}")

            # Print training and validation metrics
            print(
                f"Epoch [{epoch+1}/{self.epochs}] - "
                f"Train Loss: {train_loss:.5f} - "
                f"Val Loss: {val_loss:.5f}"
            )

    def test(self, max_display=0):
        self.model = load(self.model, "regression", exp_name=self.name)
        self.val_loader = DataLoader(
            self.val_single_dataset, batch_size=1, shuffle=False
        )
        y_true, y_pred = self.collect(
            self.val_loader, self.model, self.device, max_display=max_display
        )
        results = standard_regression_evaluation(
            np.array(y_true), np.array(y_pred), self.name, 0, self.max_days
        )
        return results[self.name][self.name + "_regression"][
            self.name + "_regression_mae"
        ]

    def init_parameters(self):
        self.name = self.config.name
        self.max_age = self.config.dataset.max_age
        self.max_days = 365 * self.max_age
        self.dataset_metadata_path = os.path.join(
            self.config.dataset.basepath, self.config.dataset.metadata
        )
        self.dataset_images_path = os.path.join(
            self.config.dataset.basepath, self.config.dataset.images
        )

        self.learning_rate = self.config.training.learning_rate
        self.batch_size = self.config.training.batch_size
        self.epochs = self.config.training.epochs
        self.train_ratio = self.config.dataset.train_ratio
        self.in_mem = self.config.dataset.in_memory
        self.resume = self.config.resume

        self.regression_lin_start = self.config.model.regression_lin_start
        self.regression_stages = self.config.model.regression_stages

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
