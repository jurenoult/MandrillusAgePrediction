import os
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.classification import Accuracy
from mandrillage.dataset import (
    ClassificationMandrillImageDataset,
    read_dataset,
    resample,
)
from mandrillage.evaluations import standard_classification_evaluation
from mandrillage.models import CoralModel, VGGFace
from mandrillage.pipeline import Pipeline
from mandrillage.utils import load, save, split_indices, create_kfold_data, softmax

from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.losses import coral_loss


def np_proba_to_label(probas):
    predict_levels = probas > 0.5
    predicted_labels = np.sum(predict_levels, axis=-1)
    return predicted_labels


log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class CoralClassificationPipeline(Pipeline):
    def __init__(self):
        super(CoralClassificationPipeline, self).__init__()

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

        # self.data = resample(self.data, bins=self.n_classes)

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
        self.train_dataset = ClassificationMandrillImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=self.in_mem,
            days_step=self.max_days / self.n_classes,
            n_classes=self.n_classes,
            individuals_ids=self.train_indices,
            return_integer=True,
        )

        self.val_dataset = ClassificationMandrillImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=self.in_mem,
            days_step=self.max_days / self.n_classes,
            n_classes=self.n_classes,
            individuals_ids=self.val_indices,
            return_integer=False,
        )

        self.train_loader = self.make_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self.make_dataloader(self.val_dataset)

    def init_logging(self):
        pass

    def init_model(self):
        self.backbone = VGGFace(
            start_filters=self.vgg_start_filters, output_dim=self.vgg_output_dim
        )
        self.model = CoralModel(
            self.backbone,
            input_dim=self.backbone.output_dim,
            num_classes=self.n_classes,
        )
        self.backbone = self.backbone.to(self.device)
        self.model = self.model.to(self.device)

    def init_losses(self):
        # Losses
        self.criterion = coral_loss
        self.val_criterion = self.val_criterion = Accuracy(
            task="multiclass", num_classes=self.n_classes
        ).to(self.device)

    def init_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def init_callbacks(self):
        pass

    def init_loggers(self):
        pass

    def train_step(self, loader, model, criterion, device):
        x, y = next(iter(loader))

        y = levels_from_labelbatch(y, num_classes=self.n_classes)

        x, y = self.xy_to_device(x, y, device)

        # Forward pass
        logits, _ = model(x)
        loss = criterion(logits, y)

        return loss, self.get_size(x)

    def train(self):
        steps = len(self.train_loader)

        if self.resume:
            self.model = load(
                self.model,
                f"coral_classification_{self.train_index}",
                exp_name=self.name,
                output_dir=self.output_dir,
            )

        # Training loop
        best_val = 0.0
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.model.train()  # Set the model to train mode
            train_loss = 0.0

            for i in tqdm(range(steps), leave=True):
                classif_loss, reg_size = self.train_step(
                    self.train_loader,
                    self.model,
                    self.criterion,
                    self.device,
                )

                loss = classif_loss
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += classif_loss.item() * reg_size

                n_samples = self.batch_size * (i + 1)
                pbar.set_description(
                    f"Classification train coral loss: {(train_loss/n_samples):.5f}"
                )

            train_loss /= len(self.train_dataset)

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            val_loss = 0.0

            with torch.no_grad():
                val_loss = self.val_loss(
                    self.val_loader, self.model, self.val_criterion, self.device
                )
            val_loss /= len(self.val_dataset)

            if val_loss > best_val:
                log.info(f"Val loss improved from {best_val:.4f} to {val_loss:.4f}")
                best_val = val_loss
                save(
                    self.model,
                    f"coral_classification_{self.train_index}",
                    exp_name=self.name,
                    output_dir=self.output_dir,
                )
            else:
                log.info(f"Val loss did not improved from {best_val:.4f}")

            save(
                self.model,
                f"last_coral_classification_{self.train_index}",
                exp_name=self.name,
                output_dir=self.output_dir,
            )

            # Print training and validation metrics
            log.info(
                f"Epoch [{epoch+1}/{self.epochs}] - "
                f"Train Loss: {train_loss:.5f} - "
                f"Val Loss: {val_loss:.5f}"
            )

    def val_step(self, x, y, model, criterion, device):
        x, y = self.xy_to_device(x, y, device)
        logits, probas = model(x)

        y_hat = proba_to_label(probas)
        y = torch.argmax(y, dim=-1)
        loss = criterion(y_hat, y)
        return loss.item() * self.get_size(x)

    def predict_from_dataset(self, x):
        z = torch.unsqueeze(x, axis=0)
        z = z.to(self.device)
        _, pred = self.model(z)
        pred = proba_to_label(pred).squeeze().detach().cpu().numpy()
        return pred

    def predict_per_individual(self, val_dataset):
        import matplotlib.pyplot as plt

        # For each individual
        ids = val_dataset.df.groupby(["id"])

        prediction_outputdir = os.path.join(
            self.output_dir, f"prediction_{self.train_index}"
        )
        os.makedirs(prediction_outputdir, exist_ok=True)

        for _id, group in tqdm(ids, total=len(ids)):
            individual_outputdir = os.path.join(prediction_outputdir, str(_id))
            os.makedirs(individual_outputdir, exist_ok=True)
            individual_y_true = []
            individual_y_pred = []
            for j, row in group.iterrows():
                x, y = val_dataset._getpair_from_row(row)
                y = val_dataset.to_class(y)
                y_hat = self.predict_from_dataset(x)
                x = x.detach().cpu()

                individual_y_true.append(y)
                individual_y_pred.append(y_hat)

                # Visualize the images and predictions
                plt.imshow(x.permute(1, 2, 0))
                plt.title(f"Predicted: {y_hat}, Real: {y}")
                plt.savefig(os.path.join(individual_outputdir, row["photo_name"]))
                plt.close()

            fig = plt.figure(figsize=(12, 10))
            plt.scatter(individual_y_true, individual_y_pred)
            plt.plot(individual_y_true, individual_y_true)
            plt.savefig(
                os.path.join(individual_outputdir, "growth_prediction.png"),
            )
            plt.close()

    def collect(self, loader, model, device, max_display=0):
        y_true = []
        y_pred = []

        # Perform inference on validation images
        for i, (images, targets) in enumerate(loader):
            # Forward pass
            images = images.to(device)
            outputs = model(images)

            # Convert the outputs to numpy arrays
            _, probs = outputs

            probs = probs.squeeze().detach().cpu().numpy()
            real_class = np.argmax(targets.squeeze().cpu().numpy(), axis=-1)

            y_true.append(real_class)
            y_pred.append(probs)

        return y_true, y_pred

    def test(self, max_display=0):
        self.model = load(
            self.model,
            f"coral_classification_{self.train_index}",
            exp_name=self.name,
            output_dir=self.output_dir,
        )
        self.model.eval()

        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        y_true, y_pred = self.collect(
            self.val_loader, self.model, self.device, max_display=max_display
        )

        results = standard_classification_evaluation(
            np.array(y_true),
            np.array(y_pred),
            self.max_days / self.n_classes,
            self.n_classes,
            "classification",
            np_proba_to_label,
        )

        scores_path = os.path.join(self.output_dir, f"scores_{self.train_index}.json")
        with open(scores_path, "w") as file:
            import json

            file.write(json.dumps(results, cls=NumpyEncoder))

        log.info("Performing inference per individual")
        self.predict_per_individual(self.val_dataset)

        return results[self.name][self.name + "_regression"][
            self.name + "_regression_mae"
        ]

    def init_parameters(self):
        super().init_parameters()

        self.vgg_start_filters = self.config.model.vgg_start_filters
        self.vgg_output_dim = self.config.model.vgg_output_dim
        self.regression_lin_start = self.config.model.regression_lin_start
        self.regression_stages = self.config.model.regression_stages
        self.n_classes = self.config.dataset.n_classes
