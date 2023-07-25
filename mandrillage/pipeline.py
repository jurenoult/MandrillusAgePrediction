import logging
import torch
import os
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)


class Pipeline(object):
    def __init__(self) -> None:
        torch.manual_seed(0)

    def set_config(self, config):
        OmegaConf.resolve(config)
        self.config = config
        self.init_parameters()

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device:", self.device.type)

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

    def val_loss(self, loader, model, criterion, device):
        total_val_loss = 0.0
        for x, y in tqdm(loader, leave=True):
            total_val_loss += self.val_step(x, y, model, criterion, device)
        return total_val_loss

    def val_step(self, x, y, model, criterion, device):
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        return loss.item() * x.size(0)

    def train_step(self, loader, optimizer, model, criterion, device):
        x, y = next(iter(loader))
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = x[i].to(device)
        else:
            x = x.to(device)
        optimizer.zero_grad()

        # Forward pass
        y = y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if isinstance(x, list):
            size = x[0].size(0)
        else:
            size = x.size(0)

        return loss.item() * size

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
            plt.title(f"Predicted: {pred}, Actual: {target}")
            plt.show()

        return y_true, y_pred

    def test(self):
        raise ValueError("You must subclass self.test() method")

    def init(self):
        self.init_logging()
        self.init_datamodule()
        self.init_model()
        self.init_losses()
        self.init_optimizers()
        self.init_callbacks()
        self.init_loggers()

    def run(self):
        assert (
            self.config.train or self.config.test
        ), "At least train or test must be true"

        self.init()
        training_score = None
        test_score = None

        if self.config.train:
            training_score = self.train()
        if self.config.test:
            test_score = self.test()

        self.score = training_score
        if self.config.test:
            self.score = test_score

        log.info(f"Final score : {self.score}")

        return self.score
