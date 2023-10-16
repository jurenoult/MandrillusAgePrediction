import os
import json
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
import mlflow

from mandrillage.dataset import (
    MandrillImageDataset,
    read_dataset,
    MandrillSimilarityImageDataset,
    AugmentedDataset,
    AugmentedSimilarityDataset,
)
from mandrillage.evaluations import standard_regression_evaluation
from mandrillage.models import SequentialModel
from mandrillage.pipeline import Pipeline
from mandrillage.display import display_predictions
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
        self.sim_model = None

    def make_dataloader(self, dataset, shuffle=False, sampler=None):
        if sampler:
            return DataLoader(
                dataset,
                batch_sampler=sampler,
            )
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
            sex=self.sex,
        )

        # Make the split based on individual ids (cannot separate photos from the same id)
        if self.kfold == 0:
            self.train_indices, self.val_indices = split_indices(self.data, self.train_ratio)
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
            training=True,
            normalize_y=self.config.dataset.normalize_y,
        )

        if self.config.training.use_augmentation:
            self.train_dataset = AugmentedDataset(self.train_dataset)

        self.train_similarity_dataset = MandrillSimilarityImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=False,
            max_days=self.max_days,
            individuals_ids=self.train_indices,
        )
        self.train_similarity_dataset.set_images(self.train_dataset.images)

        if self.config.training.use_similarity_augmentation:
            self.train_similarity_dataset = AugmentedSimilarityDataset(
                self.train_similarity_dataset
            )

        self.val_dataset = MandrillImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=self.in_mem,
            max_days=self.max_days,
            individuals_ids=self.val_indices,
            normalize_y=self.config.dataset.normalize_y,
        )

        self.train_loader = self.make_dataloader(self.train_dataset, shuffle=True)
        self.train_similarity_loader = self.make_dataloader(
            self.train_similarity_dataset, shuffle=True
        )
        self.val_loader = self.make_dataloader(self.val_dataset)

    def update_from_ages_steps(self, ages_steps, epoch):
        if epoch in ages_steps:
            max_age = ages_steps[epoch]
            log.info(f"Setting training max age to {max_age}")
            self.train_dataset.filter_by_age(max_age)
            self.train_loader = self.make_dataloader(self.train_dataset, shuffle=True)

    def init_logging(self):
        pass

    def init_model(self):
        self.backbone = hydra.utils.instantiate(self.config.backbone)

        if self.config.training.backbone_checkpoint:
            if os.path.exists(self.config.training.backbone_checkpoint):
                try:
                    self.backbone.load_weights(self.config.training.backbone_checkpoint)
                    log.info(
                        f"Backbone weights loaded from {self.config.training.backbone_checkpoint}"
                    )
                except Exception as err:
                    log.error(err)
                    log.error(
                        f"Failed to load backbone weights directly from {self.config.training.backbone_checkpoint}"
                    )
                    try:
                        model = torch.load(self.config.training.backbone_checkpoint)
                        self.backbone = model.backbone
                        log.info(
                            f"Backbone loaded from full model at {self.config.training.backbone_checkpoint}"
                        )
                    except Exception:
                        log.error("Failed to load backbone model")
            else:
                log.error(
                    f"Could not find model path at {self.config.training.backbone_checkpoint}"
                )

        self.config.regression_head.input_dim = self.backbone.output_dim
        self.regression_head = hydra.utils.instantiate(self.config.regression_head)
        print(self.regression_head)

        self.model = SequentialModel(backbone=self.backbone, head=self.regression_head)

        if self.config.similarity_head:
            self.config.similarity_head.input_dim = self.backbone.output_dim
            self.sim_model = hydra.utils.instantiate(self.config.similarity_head)
            self.sim_model.backbone = self.backbone

        self.backbone = self.backbone.to(self.device)
        self.model = self.model.to(self.device)
        if self.sim_model:
            self.sim_model = self.sim_model.to(self.device)

    def init_losses(self):
        train_error_function = hydra.utils.instantiate(self.config.train_regression_loss)
        val_error_function = hydra.utils.instantiate(self.config.val_regression_loss)

        if self.sim_model:
            self.sim_criterion = hydra.utils.instantiate(self.config.train_similarity_loss)

        self.criterion = train_error_function
        self.val_criterions = {
            "L1": val_error_function,
        }
        self.watch_val_loss = "L1"

    def init_optimizers(self):
        if not self.config.training.train_backbone:
            # Freeze backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

        all_parameters = []
        all_parameters += list(self.model.parameters())
        if self.sim_model:
            all_parameters += list(self.sim_model.parameters())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def init_callbacks(self):
        pass

    def init_loggers(self):
        pass

    def mean_std(self, loader, model, device):
        predictions = {}
        days_scale = self.max_days if self.config.dataset.normalize_y else 1
        xs = []
        for x_batch, y_batch in tqdm(loader, leave=True):
            x_batch = x_batch.to(device)
            y_pred_batch = model(x_batch)
            for i in range(x_batch.shape[0]):
                xs.append(x_batch[i])
                y, y_pred = y_batch[i], y_pred_batch[i]
                y = y.detach().cpu().numpy() * days_scale
                if y not in predictions:
                    predictions[y] = []
                predictions[y].append(y_pred.detach().cpu().numpy() * days_scale)

        std_by_value = {}
        mean_by_value = {}
        for y, values in predictions.items():
            std_by_value[y] = np.std(np.array(values))
            mean_by_value[y] = np.mean(np.array(values))

        fig = display_predictions(
            predictions,
            std_by_value,
            mean_by_value,
            os.path.join(self.output_dir, "latest_val_performance"),
        )

        return np.mean(list(std_by_value.values())), fig

    def save_best_val_loss(self, val_loss, best_val):
        if val_loss < best_val:
            log.info(
                f"Val loss {self.watch_val_loss} improved from {best_val:.4f} to {val_loss:.4f}"
            )
            best_val = val_loss
            save(
                self.model,
                f"regression_{self.train_index}",
                exp_name=self.name,
                output_dir=self.output_dir,
            )
        else:
            log.info(f"Val loss did not improved from {best_val:.4f}")
        return best_val

    def sim_train_step(
        self,
    ):
        x1, x2, y = next(
            iter(
                self.train_similarity_loader,
            )
        )
        y = y.to(self.device)
        x1, x2 = self.xy_to_device(x1, x2, self.device)
        y_pred = self.sim_model((x1, x2))
        sim_loss = self.sim_criterion(y_pred, y)
        return sim_loss, self.get_size(x1)

    def train(self):
        if self.resume:
            self.model = load(
                self.model,
                f"regression_{self.train_index}",
                exp_name=self.name,
                output_dir=self.output_dir,
            )

        # Age steps with number of epochs

        # This mean that we set the max age at time 0 to the global max age
        # It is the default behavior
        epoch_step = self.epochs
        age_step = self.max_age

        # We increase the max age of both train/val dataset incrementally
        # epoch_step = 10
        # age_step = 0.2

        ages_steps = {
            i * epoch_step: min(self.max_age, (i + 1) * age_step)
            for i in range(int(self.max_age // age_step) + 1)
        }
        print(ages_steps)

        # Training loop
        best_val = np.inf
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.update_from_ages_steps(ages_steps, epoch)

            self.model.train()  # Set the model to train mode
            if self.sim_model:
                self.sim_model.train()
            train_loss = 0.0
            train_sim_loss = 0.0

            for i in tqdm(range(len(self.train_loader)), leave=True):
                self.optimizer.zero_grad()

                reg_loss, reg_size = self.train_step(
                    self.train_loader,
                    self.optimizer,
                    self.model,
                    self.criterion,
                    self.device,
                )

                # SIMILARITY LOSS
                if self.config.training.sim_weight > 0 and self.sim_model:
                    sim_loss, sim_size = self.sim_train_step()
                    loss = (
                        self.config.training.reg_weight * reg_loss
                        + self.config.training.sim_weight * sim_loss
                    )
                    train_sim_loss += sim_loss.item() * sim_size
                else:
                    loss = reg_loss

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                train_loss += reg_loss.item() * reg_size

                n_samples = self.batch_size * (i + 1)
                train_description_str = f"Regression train loss: \
                    {(train_loss/n_samples):.5f} -\
                    Similarity train loss: {(train_sim_loss/n_samples):.5f}"
                pbar.set_description(train_description_str)
            log.info(train_description_str)
            train_loss /= len(self.train_dataset)

            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            val_losses = {}

            with torch.no_grad():
                for val_name, val_fnct in self.val_criterions.items():
                    val_losses[val_name] = self.val_loss(
                        self.val_loader, self.model, val_fnct, self.device
                    )
                    val_losses[val_name] /= len(self.val_dataset)

                # Add mean std
                mean_std, fig = self.mean_std(
                    loader=self.val_loader, model=self.model, device=self.device
                )
                val_losses["mean_std"] = mean_std
                # mlflow.log_figure(fig, "mean_std")

            val_loss = val_losses[self.watch_val_loss]
            best_val = self.save_best_val_loss(val_loss, best_val)

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_loss_std", val_losses["mean_std"], step=epoch)
            mlflow.log_metric("best_val_loss", best_val, step=epoch)

            # Print training and validation metrics
            val_str = " - ".join(
                [f" val_{name}: {value:.5f}" for name, value in val_losses.items()]
            )
            log.info(
                f"Epoch [{epoch+1}/{self.epochs}] - "
                f"Train Loss: {train_loss:.5f} - "
                f"{val_str}"
            )

            # Save last iteration
            save(
                self.model,
                f"regression_{self.train_index}_last",
                exp_name=self.name,
                output_dir=self.output_dir,
            )
        return best_val

    def predict_from_dataset(self, x):
        z = torch.unsqueeze(x, axis=0)
        z = z.to(self.device)
        outputs = self.model(z)
        pred = outputs.squeeze().detach().cpu().numpy()
        return pred

    def predict_per_individual(self, val_dataset):
        import matplotlib.pyplot as plt

        # For each individual
        ids = val_dataset.df.groupby(["id"])

        prediction_outputdir = os.path.join(self.output_dir, f"prediction_{self.train_index}")
        os.makedirs(prediction_outputdir, exist_ok=True)

        days_scale = self.max_days if self.config.dataset.normalize_y else 1

        for _id, group in tqdm(ids):
            individual_outputdir = os.path.join(prediction_outputdir, str(_id))
            os.makedirs(individual_outputdir, exist_ok=True)
            individual_y_true = []
            individual_y_pred = []
            for j, row in tqdm(group.iterrows(), leave=True):
                x, y = val_dataset._getpair_from_row(row)
                y_hat = self.predict_from_dataset(x)
                y = y.detach().cpu().numpy()
                x = x.detach().cpu()

                y = y * days_scale
                y_hat = y_hat * days_scale

                individual_y_true.append(y)
                individual_y_pred.append(y_hat)

                # Visualize the images and predictions
                plt.imshow(x.permute(1, 2, 0))
                plt.title(f"Predicted: {y_hat}, Real: {y}, Error: {abs(y_hat-y)}")
                plt.savefig(os.path.join(individual_outputdir, row["photo_name"]))
                plt.close()

            y_individual_pred_means = []
            np_pred = np.array(individual_y_pred)
            np_true = np.array(individual_y_true)
            for y_true in individual_y_true:
                mean = np.mean(np_pred[np_true == y_true])
                y_individual_pred_means.append(mean)

            plt.figure(figsize=(12, 10))

            plt.xlim([0, self.max_days])
            plt.ylim([0, self.max_days])
            plt.xlabel("Real age (in days)")
            plt.ylabel("Predicted age (in days)")
            plt.scatter(individual_y_true, individual_y_pred)
            plt.scatter(individual_y_true, y_individual_pred_means, marker="^")
            plt.plot(individual_y_true, individual_y_true)
            plt.savefig(
                os.path.join(individual_outputdir, "growth_prediction.png"),
            )
            plt.close()

    def test(self, max_display=0):
        self.model = load(
            self.model,
            f"regression_{self.train_index}",
            exp_name=self.name,
            output_dir=self.output_dir,
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

        # log.info("Performing inference per individual")
        # self.predict_per_individual(self.val_dataset)

        return results[self.name][self.name + "_regression"][self.name + "_regression_mae"]

    def init_parameters(self):
        super().init_parameters()
