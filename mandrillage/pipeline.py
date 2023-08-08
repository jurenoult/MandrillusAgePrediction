import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)


class Pipeline(object):
    def __init__(self) -> None:
        torch.manual_seed(0)
        self.train_index = 0

    def set_config(self, config, output_dir):
        OmegaConf.resolve(config)
        self.config = config
        self.output_dir = output_dir
        self.init_parameters()

    def init_parameters(self):
        self.name = self.config.name
        self.max_age = self.config.dataset.max_age
        self.max_days = 365 * self.max_age
        self.max_dob_error = self.config.dataset.max_dob_error
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

        self.kfold = self.config.kfold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")

    def init_datamodule(self):
        raise ValueError("You must subclass self.init_datamodule() method")

    def init_logging(self):
        raise ValueError("You must subclass self.init_logging() method")

    def init_model(self):
        raise ValueError("You must subclass self.init_model() method")

    def init_losses(self):
        raise ValueError("You must subclass self.init_losses() method")

    def init_optimizers(self):
        raise ValueError("You must subclass self.init_optimizers() method")

    def init_callbacks(self):
        raise ValueError("You must subclass self.init_callback() method")

    def init_loggers(self):
        raise ValueError("You must subclass self.init_loggers() method")

    def train(self):
        raise ValueError("You must subclass self.train() method")

    def val_loss(self, loader, model, criterion, device, repeat=1):
        total_val_loss = 0.0

        for i in range(repeat):
            for x, y in tqdm(loader, leave=True):
                total_val_loss += self.val_step(x, y, model, criterion, device)
        return total_val_loss / repeat

    def to_device(self, x, device):
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = x[i].to(device)
        else:
            x = x.to(device)
        return x

    def xy_to_device(self, x, y, device):
        return self.to_device(x, device), self.to_device(y, device)

    def val_step(self, x, y, model, criterion, device):
        x, y = self.xy_to_device(x, y, device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        return loss.item() * self.get_size(x)

    def get_size(self, x):
        if isinstance(x, list):
            size = x[0].size(0)
        else:
            size = x.size(0)
        return size

    def train_step(self, loader, optimizer, model, criterion, device):
        x, y = next(iter(loader))
        x, y = self.xy_to_device(x, y, device)
        optimizer.zero_grad()

        # Forward pass
        y_hat = model(x)
        loss = criterion(y_hat, y)

        size = self.get_size(x)

        return loss, size

    def collect(self, loader, model, device, max_display=0):
        y_true = []
        y_pred = []

        # Perform inference on validation images
        for i, (images, targets) in enumerate(loader):
            # Forward pass
            images = images.to(device)
            outputs = model(images)

            # Convert the outputs to numpy arrays
            pred = outputs.squeeze().detach().cpu().numpy() * 365
            target = targets.squeeze().cpu().numpy() * 365

            y_true.append(target)
            y_pred.append(pred)

            if i >= max_display:
                continue

            # Display the results
            print("Predicted Values:", pred)
            print("Actual Values:", target)
            print("Prediction Error: ", pred - target)
            print()  # Add an empty line for separation

            # Visualize the images and predictions
            plt.imshow(images.squeeze().cpu().permute(1, 2, 0))
            plt.title(f"Predicted: {pred}, Actual: {target}, Error: {abs(target-pred)}")
            plt.show()

        return y_true, y_pred

    def test(self):
        raise ValueError("You must subclass self.test() method")

    def init(self):
        log.info("Initializing experiment...")
        log.info("Initializing logging...")
        self.init_logging()
        log.info("Initializing model...")
        self.init_model()
        log.info("Initializing losses...")
        self.init_losses()
        log.info("Initializing optimizers...")
        self.init_optimizers()
        log.info("Initializing callbacks...")
        self.init_callbacks()
        log.info("Initializing loggers...")
        self.init_loggers()
        log.info("Initializing datamodule...")
        self.init_datamodule()

    def run(self):
        assert (
            self.config.train or self.config.test
        ), "At least train or test must be true"

        training_score = None
        test_score = None

        if self.kfold > 1:
            test_scores = {}
            for i in range(self.kfold):
                self.train_index = i
                # Set data for this k
                self.init()
                # Train
                training_score = self.train()
                # Evaluate
                test_score = self.test()
                test_scores[i] = test_score

            # Save cross val results
            cross_val_scores_path = os.path.join(
                self.output_dir, "cross_val_scores.json"
            )
            with open(cross_val_scores_path, "w") as f:
                import json

                f.write(json.dumps(test_scores))

            # Return mean score
            test_scores = np.mean(test_scores)
        else:
            self.init()
            if self.config.train:
                training_score = self.train()
            if self.config.test:
                test_score = self.test()

        self.score = training_score
        if self.config.test:
            self.score = test_score

        log.info(f"Final score : {self.score}")

        return self.score
