import os
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
import mlflow
import pandas as pd

from lion_pytorch import Lion

from mandrillage.dataset import (
    MandrillImageDataset,
    read_dataset,
    MandrillSimilarityImageDataset,
    AugmentedDataset,
    AugmentedSimilarityDataset,
    filter_by_age,
    filter_by_quality,
    filter_by_faceview,
)
from mandrillage.evaluations import (
    standard_regression_evaluation,
    compute_cumulative_scores,
    compute_std,
)
from mandrillage.models import SequentialModel
from mandrillage.pipeline import Pipeline
from mandrillage.display import display_worst_regression_cases
from mandrillage.utils import (
    load,
    split_indices,
    create_kfold_data,
    DAYS_IN_YEAR,
    write_results,
)

log = logging.getLogger(__name__)


class BasicRegressionPipeline(Pipeline):
    def __init__(self):
        super(BasicRegressionPipeline, self).__init__()
        self.sim_model = None

    def make_dataloader(self, dataset, shuffle=False, sampler=None, is_training=False):
        num_workers = 0 if not is_training else self.config.training.max_workers
        if sampler:
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=True if num_workers > 0 else False,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
        )

    def prepare_data(self):
        # Read data
        self.data = read_dataset(
            self.dataset_metadata_path,
            filter_dob_error=True,
            filter_certainty=self.config.dataset.dob_certain_only,
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

        # Extract training set and validation set
        self.data_train = self.data[self.data["id"].isin(self.train_indices)]
        self.data_val = self.data[self.data["id"].isin(self.val_indices)]
        self.data_train.reset_index(drop=True, inplace=True)
        self.data_val.reset_index(drop=True, inplace=True)

        # Filter by age
        self.data_train = filter_by_age(
            self.data_train, age_in_days=self.train_max_age * DAYS_IN_YEAR
        )
        self.data_val = filter_by_age(self.data_val, age_in_days=self.val_max_age * DAYS_IN_YEAR)

        # Filter by photo quality
        self.data_train = filter_by_quality(self.data_train, min_quality=self.train_min_quality)
        self.data_val = filter_by_quality(self.data_val, min_quality=self.val_min_quality)

        # Filter by face orientation
        if self.train_faceview is not None:
            self.data_train = filter_by_faceview(self.data_train, faceview_type=self.train_faceview)
        if self.val_faceview is not None:
            self.data_val = filter_by_faceview(self.data_val, faceview_type=self.val_faceview)

        self.data_train.reset_index(drop=True, inplace=True)
        self.data_val.reset_index(drop=True, inplace=True)

    def init_datamodule(self):
        self.prepare_data()

        log.info("Building training dataset...")
        self.train_dataset = MandrillImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data_train,
            img_size=self.img_size,
            in_mem=self.in_mem,
            max_days=self.max_days,
            training=True,
            normalize_y=self.config.dataset.normalize_y,
        )

        if self.config.training.use_augmentation:
            self.train_dataset = AugmentedDataset(self.train_dataset)

        self.train_similarity_dataset = MandrillSimilarityImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data_train,
            img_size=self.img_size,
            in_mem=False,
            max_days=self.max_days,
            training=True,
            normalize_y=self.config.dataset.normalize_y,
        )
        self.train_similarity_dataset.set_images(self.train_dataset.images)

        if self.config.training.use_similarity_augmentation:
            self.train_similarity_dataset = AugmentedSimilarityDataset(
                self.train_similarity_dataset
            )

        log.info("Building validation dataset...")
        self.val_dataset = MandrillImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data_val,
            img_size=self.img_size,
            in_mem=self.in_mem,
            max_days=self.max_days,
            normalize_y=self.config.dataset.normalize_y,
        )

        self.train_loader = self.make_dataloader(self.train_dataset, shuffle=True, is_training=True)
        self.train_similarity_loader = self.make_dataloader(
            self.train_similarity_dataset, shuffle=True
        )
        self.val_loader = self.make_dataloader(self.val_dataset)

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
            "MSE": torch.nn.MSELoss(),
            "L1": val_error_function,
        }
        self.watch_val_loss = "L1"

    def init_optimizers(self):
        if not self.config.training.train_backbone:
            # Freeze backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

        parameters = []
        parameters += list(self.model.parameters())
        if self.sim_model:
            parameters += list(self.sim_model.parameters())

        if self.config.training.optimizer == "lion":
            self.optimizer = Lion(parameters, lr=self.learning_rate, weight_decay=1e-2)
        elif self.config.training.optimizer == "adam":
            self.optimizer = optim.AdamW(parameters, lr=self.learning_rate, weight_decay=1e-2)

    def init_callbacks(self):
        pass

    def init_loggers(self):
        pass

    def sim_train_step(
        self,
    ):
        x1, x2, y = next(
            iter(
                self.train_similarity_loader,
            )
        )
        y = y.to(self.device)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        y_pred = self.sim_model((x1, x2))
        sim_loss = self.sim_criterion(y_pred, y)
        return sim_loss

    def collect_data(self, dataset, model, min_size_predict=256):
        y_true = []
        y_pred = []
        metadatas = []
        photo_paths = []
        errors = []
        prediction_stack = []
        n = len(dataset)
        for i in tqdm(range(n), leave=False):
            x, y = dataset[i]
            x = x.to(self.device)
            metadata = dataset.get_metadata_at_index(i)

            prediction_stack.append(x)
            if len(prediction_stack) >= min_size_predict or i == n - 1:
                prediction_stack = torch.stack(prediction_stack)
                predictions = model(prediction_stack)

                for j in range(predictions.shape[0]):
                    prediction = predictions[j].detach().cpu()
                    prediction = np.round(prediction.numpy() * self.days_scale)
                    y_pred.append(prediction)

                prediction_stack = []

            y = np.round(y.numpy() * self.days_scale)
            y_true.append(y)
            metadatas.append(metadata["id"])
            photo_paths.append(metadata["photo_path"])

        assert len(y_pred) == len(y_true)

        errors = [abs(y - y_hat) for y, y_hat in zip(y_true, y_pred)]

        df = pd.DataFrame(
            {
                "y_pred": y_pred,
                "y_true": y_true,
                "id": metadatas,
                "photo_path": photo_paths,
                "error": errors,
            }
        )

        return df

    def val_loss_from_df(self, df, loss):
        y_true = torch.tensor(np.array(list(df["y_true"])))
        y_true = torch.unsqueeze(y_true, axis=-1)
        y_pred = torch.tensor(np.array(list(df["y_pred"])))
        y_pred = torch.unsqueeze(y_pred, axis=-1)
        return loss(y_pred, y_true)

    def update_from_ages_steps(self, ages_steps, epoch):
        if epoch in ages_steps and epoch != 0:
            max_age = ages_steps[epoch]
            log.info(f"Setting training max age to {max_age}")
            self.train_dataset.filter_by_age(max_age)
            self.train_loader = self.make_dataloader(self.train_dataset, shuffle=True)

    def train_mode(self):
        self.model.train()
        if self.sim_model:
            self.sim_model.train()

    def train_step(self):
        loss = self.basic_train_step(
            self.train_loader,
            self.optimizer,
            self.model,
            self.criterion,
            self.device,
        )

        # SIMILARITY LOSS
        if self.config.training.sim_weight > 0 and self.sim_model:
            sim_loss = self.sim_train_step()
            loss = (
                self.config.training.reg_weight * loss + self.config.training.sim_weight * sim_loss
            )
        return loss

    def validate_epoch(self, epoch):
        self.model.eval()
        val_losses = {}

        with torch.no_grad():
            # Predict on validation dataset sample per sample to get metadata
            val_df = self.collect_data(self.val_dataset, self.model)

            for val_name, val_fnct in self.val_criterions.items():
                val_losses[val_name] = self.val_loss_from_df(val_df, val_fnct)

            std_by_age, std_by_age_by_id = compute_std(val_df, self.output_dir)
            # Add mean std
            val_losses["mean_std"] = np.mean(list(std_by_age.values()))
            # Add mean std per id per age
            val_losses["mean_std_by_id_by_age"] = np.mean(list(std_by_age_by_id.values()))

            display_worst_regression_cases(
                val_df, self.val_dataset, self.days_scale, self.output_dir, max_n=10, epoch=epoch
            )

            css = compute_cumulative_scores(val_df)

        val_losses["MSE"] = val_losses["MSE"] / (DAYS_IN_YEAR**2)
        val_loss = val_losses[self.watch_val_loss]

        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_loss_mse", val_losses["MSE"], step=epoch)
        mlflow.log_metric("val_loss_std", val_losses["mean_std"], step=epoch)
        mlflow.log_metric("val_loss_std_by_id", val_losses["mean_std_by_id_by_age"], step=epoch)
        mlflow.log_metrics(css, step=epoch)

        return val_loss, val_df, val_losses

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

                y = np.round(y * self.days_scale)
                y_hat = np.round(y_hat * self.days_scale)

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
        # Load
        self.model = load(
            self.model,
            f"regression_{self.train_index}",
            exp_name=self.name,
            output_dir=self.output_dir,
        )
        self.model.eval()

        data = self.collect_data(self.val_dataset, self.model)
        y_true = data["y_true"]
        y_pred = data["y_pred"]

        results = standard_regression_evaluation(
            np.array(y_true), np.array(y_pred), self.name, 0, self.max_days
        )

        scores_path = os.path.join(self.output_dir, f"scores_{self.train_index}.json")
        write_results(scores_path, results)

        # log.info("Performing inference per individual")
        # self.predict_per_individual(self.val_dataset)

        return results[self.name][self.name + "_regression"][self.name + "_regression_mae"]

    def init_parameters(self):
        super().init_parameters()
