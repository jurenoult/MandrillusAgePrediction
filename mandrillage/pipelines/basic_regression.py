import os
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mandrillage.dataset import (
    MandrillImageDataset,
    read_dataset,
    MandrillSimilarityImageDataset,
    resample,
    AugmentedDataset,
    AugmentedSimilarityDataset,
)
from mandrillage.evaluations import standard_regression_evaluation
from mandrillage.models import RegressionModel, VGGFace, VoloBackbone, CoAtNetBackbone
from mandrillage.pipeline import Pipeline
from mandrillage.utils import load, save, split_indices, create_kfold_data
from mandrillage.losses import (
    BMCLoss,
    ScalerLoss,
    LinearWeighting,
    AdaptiveMarginLoss,
    FeatureSimilarityLoss,
)

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
            sex=self.sex,
        )

        print("Dataset length")
        print(len(self.data))
        input()

        # self.data = resample(self.data, bins=int(self.max_age))

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
            training=True,
        )

        self.train_dataset = AugmentedDataset(self.train_dataset)

        self.train_similarity_dataset = MandrillSimilarityImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=False,
            max_days=self.max_days,
            individuals_ids=self.train_indices,
        )
        self.train_similarity_dataset.set_images(self.train_dataset.images)

        self.train_similarity_dataset = AugmentedSimilarityDataset(
            self.train_similarity_dataset
        )

        self.val_dataset = MandrillImageDataset(
            root_dir=self.dataset_images_path,
            dataframe=self.data,
            in_mem=self.in_mem,
            max_days=self.max_days,
            individuals_ids=self.val_indices,
        )

        self.train_loader = self.make_dataloader(self.train_dataset, shuffle=True)
        self.train_similarity_loader = self.make_dataloader(
            self.train_similarity_dataset, shuffle=True
        )
        self.val_loader = self.make_dataloader(self.val_dataset)

    def init_logging(self):
        pass

    def init_model(self):
        self.backbone = VGGFace(
            start_filters=self.vgg_start_filters, output_dim=self.vgg_output_dim
        )
        if self.vgg_face_pretrained_path:
            if os.path.exists(self.vgg_face_pretrained_path):
                self.backbone.load_weights(self.vgg_face_pretrained_path)
            else:
                log.error(
                    f"Could not find model path at {self.vgg_face_pretrained_path}"
                )

        # self.backbone = CoAtNetBackbone(output_dim=1024)
        self.model = RegressionModel(
            self.backbone,
            input_dim=self.backbone.output_dim,
            lin_start=self.regression_lin_start,
            n_lin=self.regression_stages,
        )
        self.backbone = self.backbone.to(self.device)
        self.model = self.model.to(self.device)

    def init_losses(self):
        # train_error_function = nn.MSELoss()
        # val_error_function = nn.L1Loss()
        days = 10
        min_days = days
        max_days = days

        train_error_function = AdaptiveMarginLoss(
            min_days_error=min_days, max_days_error=max_days, max_days=self.max_days
        )
        self.sim_criterion = nn.L1Loss()
        val_error_function = nn.L1Loss()
        val_error_function_margin = AdaptiveMarginLoss(
            min_days_error=min_days, max_days_error=max_days, max_days=self.max_days
        )

        self.criterion = train_error_function
        self.val_criterions = {
            "L1": val_error_function,
            "marginloss": val_error_function_margin,
        }
        self.watch_val_loss = "L1"

        # Losses
        if self.linear_weighting:
            self.mse_loss_weighting = LinearWeighting(
                min_error=self.linear_weighting.min_age_error_max,
                max_error=self.linear_weighting.max_age_error_max,
                max_days=self.max_days,
                error_function=train_error_function,
            )
            self.criterion = ScalerLoss(train_error_function, self.mse_loss_weighting)

            self.l1_loss_weighting = LinearWeighting(
                min_error=self.linear_weighting.min_age_error_max,
                max_error=self.linear_weighting.max_age_error_max,
                max_days=self.max_days,
                error_function=val_error_function,
            )

            self.val_criterions = {
                "L1": val_error_function,
                "L1 scaled": ScalerLoss(val_error_function, self.l1_loss_weighting),
                "marginloss": val_error_function_margin,
            }

            self.watch_val_loss = "L1 scaled"

    def init_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if isinstance(self.criterion, BMCLoss):
            self.optimizer.add_param_group(
                {
                    "params": self.criterion.noise_sigma,
                    "lr": 1e-2,
                    "name": "noise_sigma",
                }
            )

    def init_callbacks(self):
        pass

    def init_loggers(self):
        pass

    def train(self):
        steps = len(self.train_loader)

        if self.resume:
            self.model = load(
                self.model,
                f"regression_{self.train_index}",
                exp_name=self.name,
                output_dir=self.output_dir,
            )

        # Find best lr stuff
        # find_lr = False
        # if find_lr:
        #     from torch_lr_finder import LRFinder

        #     lr_finder = LRFinder(
        #         self.model, self.optimizer, self.criterion, device=self.device
        #     )
        #     lr_finder.range_test(
        #         self.train_loader, num_iter=300, start_lr=1e-7, end_lr=100
        #     )
        #     lr_finder.plot()  # to inspect the loss-learning rate graph
        #     lr_finder.reset()  # to reset the model and optimizer to their initial state

        # Training loop
        best_val = np.inf
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.model.train()  # Set the model to train mode
            train_loss = 0.0
            train_sim_loss = 0.0

            for i in tqdm(range(steps), leave=True):
                reg_loss, reg_size = self.train_step(
                    self.train_loader,
                    self.optimizer,
                    self.model,
                    self.criterion,
                    self.device,
                )

                ### SIMILARITY LOSS
                if self.sim_alpha > 0:
                    x1, x2 = next(
                        iter(
                            self.train_similarity_loader,
                        )
                    )
                    x1, x2 = self.xy_to_device(x1, x2, self.device)
                    y1, y2 = self.model(x1), self.model(x2)
                    sim_loss = self.sim_criterion(y1, y2)

                    loss = reg_loss + self.sim_alpha * sim_loss
                    train_sim_loss += sim_loss.item() * self.get_size(x1)
                else:
                    loss = reg_loss

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                train_loss += reg_loss.item() * reg_size

                n_samples = self.batch_size * (i + 1)
                pbar.set_description(
                    f"Regression train loss: {(train_loss/n_samples):.5f} - Similarity train loss: {(train_sim_loss/n_samples):.5f}"  # - Weight slope : {self.criterion.weight.a}"
                )

            train_loss /= len(self.train_dataset)
            self.criterion.display_stored_values("train_margin")

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            val_losses = {}

            # Store values for analysis

            with torch.no_grad():
                for val_name, val_fnct in self.val_criterions.items():
                    val_losses[val_name] = self.val_loss(
                        self.val_loader, self.model, val_fnct, self.device
                    )
                    val_losses[val_name] /= len(self.val_dataset)

            val_loss = val_losses[self.watch_val_loss]
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

            # Display stored values
            self.val_criterions["marginloss"].display_stored_values("val_margin")

            # Print training and validation metrics
            val_str = " - ".join(
                [f" val_{name}: {value:.5f}" for name, value in val_losses.items()]
            )
            log.info(
                f"Epoch [{epoch+1}/{self.epochs}] - "
                f"Train Loss: {train_loss:.5f} - "
                f"{val_str}"
            )

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

        prediction_outputdir = os.path.join(
            self.output_dir, f"prediction_{self.train_index}"
        )
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

                y = int(y * self.max_days)
                y_hat = int(y_hat * self.max_days)

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

            fig = plt.figure(figsize=(12, 10))

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
        self.sim_alpha = self.config.training.sim_alpha
        self.vgg_face_pretrained_path = self.config.model.pretrained

        self.linear_weighting = self.config.training.linear_weighting
