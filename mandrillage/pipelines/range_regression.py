import os
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mandrillage.evaluations import standard_regression_evaluation
from mandrillage.models import RegressionModel, VGGFace
from mandrillage.pipelines.basic_regression import BasicRegressionPipeline
from mandrillage.losses import RangeLoss, BinaryRangeMetric
from mandrillage.utils import load, save

log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class RangeRegressionPipeline(BasicRegressionPipeline):
    def __init__(self):
        super(RangeRegressionPipeline, self).__init__()

    def init_datamodule(self):
        super().init_datamodule()

    def init_model(self):
        self.backbone = VGGFace(
            start_filters=self.vgg_start_filters, output_dim=self.vgg_output_dim
        )
        self.model = RegressionModel(
            self.backbone,
            input_dim=self.backbone.output_dim,
            output_dim=2,
            lin_start=self.regression_lin_start,
            n_lin=self.regression_stages,
        )
        self.backbone = self.backbone.to(self.device)
        self.model = self.model.to(self.device)

    def init_losses(self):
        # Losses
        self.criterion = RangeLoss()
        self.val_criterion = BinaryRangeMetric()

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
                self.model,
                "range_regression",
                exp_name=self.name,
                output_dir=self.output_dir,
            )

        # Training loop
        best_val = 0
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
                # x1, x2 = next(
                #     iter(
                #         self.train_similarity_loader,
                #     )
                # )
                # x1, x2 = self.xy_to_device(x1, x2, self.device)
                # y1, y2 = self.model(x1), self.model(x2)
                # sim_loss = self.criterion(y1, y2)

                # loss = reg_loss + sim_loss
                loss = reg_loss
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # train_sim_loss += sim_loss.item() * self.get_size(x1)
                train_loss += reg_loss.item() * reg_size

                n_samples = self.batch_size * (i + 1)
                pbar.set_description(
                    f"Regression train loss: {(train_loss/n_samples):.5f} - Similarity train loss: {(train_sim_loss/n_samples):.5f}"
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
                    "range_regression",
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

        prediction_outputdir = os.path.join(self.output_dir, "prediction")
        os.makedirs(prediction_outputdir, exist_ok=True)

        for _id, group in tqdm(ids):
            individual_outputdir = os.path.join(prediction_outputdir, str(_id))
            os.makedirs(individual_outputdir, exist_ok=True)
            individual_y_true = []
            individual_y_left = []
            individual_y_right = []
            individual_y_median = []
            for j, row in tqdm(group.iterrows(), leave=True):
                x, y = val_dataset._getpair_from_row(row)
                y_hat = self.predict_from_dataset(x)
                y = y.detach().cpu().numpy()
                x = x.detach().cpu()

                y = int(y * self.max_days)
                y_left = int(y_hat[..., 0] * self.max_days)
                y_right = y_left + int(y_hat[..., 1] * self.max_days)

                median_value = (y_right + y_left) / 2
                if len(individual_y_true) > 0 and individual_y_true[-1] == y:
                    individual_y_left[-1] = np.mean([individual_y_left[-1], y_left])
                    individual_y_right[-1] = np.mean([individual_y_right[-1], y_right])
                    individual_y_median[-1] = np.mean(
                        [individual_y_median[-1], median_value]
                    )
                else:
                    individual_y_true.append(y)
                    individual_y_left.append(y_left)
                    individual_y_right.append(y_right)
                    individual_y_median.append(median_value)

                # Visualize the images and predictions
                plt.imshow(x.permute(1, 2, 0))
                plt.title(
                    f"Predicted: [{y_left};{y_right}], Real: {y}, Median: [{median_value}]; Error: {abs(y-median_value)}"
                )
                plt.savefig(os.path.join(individual_outputdir, row["photo_name"]))
                plt.close()

            fig = plt.figure(figsize=(12, 10))
            plt.fill_between(
                individual_y_true,
                individual_y_left,
                individual_y_right,
                color="b",
                alpha=0.1,
            )
            plt.plot(individual_y_true, individual_y_true)
            plt.plot(individual_y_true, individual_y_median)
            plt.savefig(
                os.path.join(individual_outputdir, "growth_prediction.png"),
            )
            plt.close()

    def test(self, max_display=0):
        self.model = load(
            self.model,
            "range_regression",
            exp_name=self.name,
            output_dir=self.output_dir,
        )
        self.model.eval()

        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        y_true, y_pred = self.collect(
            self.val_loader, self.model, self.device, max_display=max_display
        )
        y_pred = np.array(y_pred)
        y_pred = y_pred[..., 0] + y_pred[..., 1] / 2
        results = standard_regression_evaluation(
            np.array(y_true), np.array(y_pred), self.name, 0, self.max_days
        )

        log.info("Performing inference per individual")
        self.predict_per_individual(self.val_dataset)

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
        self.vgg_output_dim = self.config.model.vgg_output_dim
        self.regression_lin_start = self.config.model.regression_lin_start
        self.regression_stages = self.config.model.regression_stages
