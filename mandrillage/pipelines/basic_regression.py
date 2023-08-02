import os
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mandrillage.dataset import MandrillImageDataset, read_dataset
from mandrillage.evaluations import standard_regression_evaluation
from mandrillage.models import RegressionModel, VGGFace
from mandrillage.pipeline import Pipeline
from mandrillage.utils import load, save, split_indices, create_kfold_data

log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BasicRegressionPipeline(Pipeline):
    def __init__(self):
        super(BasicRegressionPipeline, self).__init__()

    def make_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def init_datamodule(self):
        # Read data
        self.data = read_dataset(
            self.dataset_metadata_path,
            filter_dob_error=True,
            filter_certainty=self.config.dataset.dob_certain_only,
            max_age=self.max_days,
            max_dob_error=self.max_dob_error,
        )

        # Make the split based on individual ids (cannot separate photos from the same id)
        if self.kfold == 0:
            self.train_indices, self.val_indices = split_indices(
                self.data, self.train_ratio
            )
        else:
            self.train_indices, self.val_indices = create_kfold_data(
                self.data, k=self.kfold, fold_index=self.train_index
            )

        # Create dataset based on indices
        self.train_dataset = MandrillImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=self.in_mem,
            max_days=self.max_days,
            individuals_ids=self.train_indices,
        )
        self.val_dataset = MandrillImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=self.in_mem,
            max_days=self.max_days,
            individuals_ids=self.val_indices,
        )

        self.train_loader = self.make_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self.make_dataloader(self.val_dataset)

    def init_logging(self):
        pass

    def init_model(self):
        self.backbone = VGGFace(start_filters=self.vgg_start_filters)
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
            self.model = load(
                self.model, "regression", exp_name=self.name, output_dir=self.output_dir
            )

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
                log.info(f"Val loss improved from {best_val:.4f} to {val_loss:.4f}")
                best_val = val_loss
                save(
                    self.model,
                    "regression",
                    exp_name=self.name,
                    output_dir=self.output_dir,
                )
            else:
                log.info(f"Val loss did not improved from {best_val:.4f}")

            # Print training and validation metrics
            log.info(
                f"Epoch [{epoch+1}/{self.epochs}] - "
                f"Train Loss: {train_loss:.5f} - "
                f"Val Loss: {val_loss:.5f}"
            )

    def test(self, max_display=0):
        self.model = load(
            self.model, "regression", exp_name=self.name, output_dir=self.output_dir
        )
        self.model.eval()

        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        y_true, y_pred = self.collect(
            self.val_loader, self.model, self.device, max_display=max_display
        )
        results = standard_regression_evaluation(
            np.array(y_true), np.array(y_pred), self.name, 0, self.max_days
        )

        scores_path = os.path.join(self.output_dir, f"scores_{self.train_index}.json")
        with open(scores_path, "w") as file:
            import json

            file.write(json.dumps(results, cls=NumpyEncoder))

        return results[self.name][self.name + "_regression"][
            self.name + "_regression_mae"
        ]

    def init_parameters(self):
        super().init_parameters()

        self.vgg_start_filters = self.config.model.vgg_start_filters
        self.regression_lin_start = self.config.model.regression_lin_start
        self.regression_stages = self.config.model.regression_stages
