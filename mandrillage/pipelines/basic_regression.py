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
import pandas as pd
import matplotlib.pyplot as plt

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
from mandrillage.evaluations import standard_regression_evaluation
from mandrillage.models import SequentialModel
from mandrillage.pipeline import Pipeline
from mandrillage.display import display_predictions
from mandrillage.utils import load, save, split_indices, create_kfold_data, DAYS_IN_YEAR

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
            dataframe=self.data,
            img_size=self.img_size,
            in_mem=False,
            max_days=self.max_days,
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

        if self.config.training.optimizer == "lion":
            self.optimizer = Lion(all_parameters, lr=self.learning_rate, weight_decay=1e-2)
        elif self.config.training.optimizer == "adam":
            self.optimizer = optim.Adam(all_parameters, lr=self.learning_rate)

    def init_callbacks(self):
        pass

    def init_loggers(self):
        pass

    def compute_std(self, df, display_name="val"):
        predicted_error = {}
        predictions = {}

        # Gather data per id per age range
        for i in tqdm(range(len(df))):
            row = df.iloc[[i]]
            y_true = row["y_true"].values[0]
            y_pred = row["y_pred"].values[0]
            if y_true not in predicted_error:
                predicted_error[y_true] = {}
            id_ = row["id"].values[0]
            if id_ not in predicted_error[y_true]:
                predicted_error[y_true][id_] = []

            abs_error = abs(y_true - y_pred)
            predicted_error[y_true][id_].append(abs_error)

            if y_true not in predictions:
                predictions[y_true] = []
            predictions[y_true].append(y_pred)

        # Compute mean per id per age when multiple photo occurs
        std_by_value = {}
        std_by_value_by_id = {}
        mean_by_value = {}
        # For each unique age value
        for age in predicted_error.keys():
            age_data = predicted_error[age]
            age_pred = predictions[age]

            # Compute std per id with nb photo > 1
            age_stds_by_id = []
            for id_ in age_data.keys():
                if len(age_data[id_]) > 1:
                    current_std = np.std(np.array(age_data[id_]))
                    age_stds_by_id.append(current_std)
            age_std_by_id = np.mean(age_stds_by_id)
            if not np.isnan(age_std_by_id):
                std_by_value_by_id[age] = age_std_by_id

            # Compute std by age globally
            std_by_value[age] = np.std(np.array(age_pred))
            mean_by_value[age] = np.mean(np.array(age_pred))

        fig = display_predictions(
            predictions,
            std_by_value,
            mean_by_value,
            os.path.join(self.output_dir, f"latest_{display_name}_performance"),
        )

        return std_by_value, std_by_value_by_id

    def save_best_val_loss(self, val_loss, best_val, df):
        improved = False
        if val_loss < best_val:
            improved = True
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
            df.to_csv(os.path.join(self.output_dir, "val_raw_predictions.csv"))
        else:
            log.info(f"Val loss did not improved from {best_val:.4f}")
        return best_val, improved

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

    def display_n_worst_cases(self, df, dataset, max_n, epoch):
        n = min(max_n, len(df))

        # Make directories
        worst_cases_dir = os.path.join(self.output_dir, f"worst_{n}_cases")
        os.makedirs(worst_cases_dir, exist_ok=True)
        epoch_worst_cases_dir = os.path.join(worst_cases_dir, f"{epoch}")
        os.makedirs(epoch_worst_cases_dir, exist_ok=True)

        sorted_df = df.sort_values("error", ascending=False)

        for i in range(n):
            row = sorted_df.iloc[[i]]
            real_index = row.index.values[0]
            photo_id = row["photo_path"].values[0]

            x, y = dataset[real_index]
            y_pred = row["y_pred"].values[0]
            y = np.round(y * self.days_scale)

            plt.imshow(x.permute(1, 2, 0))
            plt.title(f"Predicted: {y_pred}, Real: {y}, Error: {abs(y - y_pred)}")
            plt.savefig(os.path.join(epoch_worst_cases_dir, f"{i}_{photo_id}.png"))
            plt.close()

    def compute_cumulative_scores(self, df):
        y_pred = np.array(list(df["y_pred"]))
        y_true = np.array(list(df["y_true"]))

        error = abs(y_pred - y_true)
        nb_values = len(y_true)

        cs_values_in_years = [1 / 12, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3]
        cs_values_in_days = [np.round(val * DAYS_IN_YEAR) for val in cs_values_in_years]

        css = {}
        for i, max_error in enumerate(cs_values_in_days):
            nb_correct = sum(error <= max_error)
            cs = float(nb_correct) / float(nb_values)
            css[f"{i}_CS_{max_error}"] = cs
        return css

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
        age_step = self.train_max_age

        # We increase the max age of both train/val dataset incrementally
        # epoch_step = 10
        # age_step = 0.2

        ages_steps = {
            i * epoch_step: min(self.train_max_age, (i + 1) * age_step)
            for i in range(int(self.train_max_age // age_step) + 1)
        }
        print(ages_steps)

        # prof = torch.profiler.profile(
        #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/regression_profiler"),
        #     record_shapes=False,
        #     with_stack=False,
        # )
        # prof.start()

        # Creates once at the beginning of training
        # Scales the gradients
        scaler = torch.cuda.amp.GradScaler()

        # Training loop
        best_val = np.inf
        pbar = tqdm(range(self.epochs))
        improved_since = 0
        for epoch in pbar:
            self.update_from_ages_steps(ages_steps, epoch)

            self.model.train()  # Set the model to train mode
            if self.sim_model:
                self.sim_model.train()
            train_loss = 0.0
            train_sim_loss = 0.0

            for i in tqdm(range(len(self.train_loader)), leave=True):
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

                # Scales the loss, and calls backward()
                # to create scaled gradients
                scaler.scale(loss).backward()

                scaler.step(self.optimizer)

                # Updates the scale for next iteration
                scaler.update()

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

            val_df = None
            with torch.no_grad():
                # Predict on validation dataset sample per sample to get metadata
                val_df = self.collect_data(self.val_dataset, self.model)

                for val_name, val_fnct in self.val_criterions.items():
                    val_losses[val_name] = self.val_loss_from_df(val_df, val_fnct)

                # Add mean std
                std_by_age, std_by_age_by_id = self.compute_std(val_df)
                val_losses["mean_std"] = np.mean(list(std_by_age.values()))
                # mlflow.log_figure(fig, "mean_std")

                # Add mean std per id per age
                val_losses["mean_std_by_id_by_age"] = np.mean(list(std_by_age_by_id.values()))

                self.display_n_worst_cases(val_df, self.val_dataset, max_n=10, epoch=epoch)

                css = self.compute_cumulative_scores(val_df)

            val_loss = val_losses[self.watch_val_loss]
            best_val, improved = self.save_best_val_loss(val_loss, best_val, val_df)

            # Early stopping
            if not improved:
                improved_since += 1
            else:
                improved_since = 0
            if improved_since >= self.early_stopping_patience:
                log.info(
                    f"Stopping training since validation loss did not improved for {self.early_stopping_patience} epochs"
                )
                break

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_loss_std", val_losses["mean_std"], step=epoch)
            mlflow.log_metric("val_loss_std_by_id", val_losses["mean_std_by_id_by_age"], step=epoch)
            mlflow.log_metric("best_val_loss", best_val, step=epoch)
            mlflow.log_metrics(css, step=epoch)

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

        # prof.stop()
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
        self.days_scale = self.max_days if self.config.dataset.normalize_y else DAYS_IN_YEAR
