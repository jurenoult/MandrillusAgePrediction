import os
import json
import logging
import hydra

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.classification import Accuracy
from mandrillage.dataset import (
    MandrillTripletDataset,
    read_dataset,
    filter_by_age
)
from mandrillage.evaluations import standard_classification_evaluation
from mandrillage.pipeline import Pipeline
from mandrillage.utils import load, save, softmax, DAYS_IN_YEAR

log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class FaceIdentificationPipeline(Pipeline):
    def __init__(self):
        super(FaceIdentificationPipeline, self).__init__()

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
            filter_unknown_dob_error=self.config.dataset.dob_error_known,
            max_dob_error=self.max_dob_error,
            sex=self.sex,
        )

        self.data = filter_by_age(
            self.data, age_in_days=self.train_max_age * DAYS_IN_YEAR
        )
        self.data.reset_index(drop=True, inplace=True)

        # Create dataset based on indices
        self.train_dataset = MandrillTripletDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=self.in_mem,
            individuals_ids=[]
        )

        self.train_loader = self.make_dataloader(self.train_dataset, shuffle=True)

    def init_logging(self):
        pass

    def init_callbacks(self):
        pass

    def init_loggers(self):
        pass

    def init_model(self):
        self.backbone = hydra.utils.instantiate(self.config.backbone)
        self.model = self.backbone
        self.model = self.model.to(self.device)

    def init_losses(self):
        # Losses
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    def init_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def train_step(self, loader, model, criterion, device):
        x1, x2, x3 = next(iter(loader))["input"]
        x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
        f1, f2, f3 = model(x1), model(x2), model(x3)
        loss = criterion(f1, f2, f3)
        return loss, x1.size(0)

    def train(self):
        steps = len(self.train_loader)

        if self.resume:
            self.model = load(
                self.model,
                f"face_identification_{self.train_index}",
                exp_name=self.name,
                output_dir=self.output_dir,
            )

        # Training loop
        pbar = tqdm(range(self.epochs))
        best_train_loss = np.inf
        for epoch in pbar:
            self.model.train()  # Set the model to train mode
            train_loss = 0.0

            for i in tqdm(range(steps), leave=True):
                face_loss, reg_size = self.train_step(
                    self.train_loader,
                    self.model,
                    self.criterion,
                    self.device,
                )

                loss = face_loss
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                train_loss += face_loss.item() * reg_size

                n_samples = self.batch_size * (i + 1)
                pbar.set_description(f"Face identification train loss: {(train_loss/n_samples):.5f}")

            train_loss /= len(self.train_dataset)

            if train_loss < best_train_loss:
                log.info(f"Train loss improved from {best_train_loss:.4f} to {train_loss:.4f}")
                best_train_loss = train_loss
                save(
                    self.model,
                    f"face_identification_{self.train_index}",
                    exp_name=self.name,
                    output_dir=self.output_dir,
                )
            else:
                log.info(f"Train loss did not improved from {best_train_loss:.4f}")

            save(
                    self.model,
                    f"face_identification_{self.train_index}_last",
                    exp_name=self.name,
                    output_dir=self.output_dir,
                )

            # Print training and validation metrics
            log.info(
                f"Epoch [{epoch+1}/{self.epochs}] - "
                f"Train Loss: {train_loss:.5f} - "
            )