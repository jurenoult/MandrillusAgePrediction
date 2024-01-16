import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import mlflow

log = logging.getLogger(__name__)

from mandrillage.utils import DAYS_IN_YEAR
from mandrillage.utils import load, save


class Pipeline(object):
    def __init__(self) -> None:
        torch.manual_seed(0)
        self.train_index = 0

    def set_config(self, config, output_dir):
        OmegaConf.resolve(config)
        self.config = config
        self.base_output_dir = output_dir
        self.init_parameters()

        exps = mlflow.search_experiments(filter_string=f"name='{self.name}'")
        if exps is None or len(exps) == 0:
            mlflow.create_experiment(self.name)

        self.current_mlflow_xp = mlflow.set_experiment(self.name)
        runs = mlflow.search_runs(experiment_names=[self.current_mlflow_xp.name])
        run_index = 0
        if runs is not None and len(runs) > 0:
            run_index = len(runs)

        self.mlflow_run = mlflow.start_run(run_name=f"{self.name}_{run_index}")
        mlflow.log_params(self.config)
        mlflow.log_param("learning_rate", self.config.training.learning_rate)
        mlflow.log_param("batch_size", self.config.training.batch_size)
        mlflow.log_param("epochs", self.config.training.epochs)
        mlflow.log_param("train_backbone", self.config.training.train_backbone)
        mlflow.log_param("use_augmentation", self.config.training.use_augmentation)
        mlflow.log_param("backbone_target", self.config.backbone._target_)
        if "n_lin" in self.config.regression_head:
            mlflow.log_param("regression_head_stages", self.config.regression_head.n_lin)
        if "lin_start" in self.config.regression_head:
            mlflow.log_param("regression_head_start_neurons", self.config.regression_head.lin_start)
        mlflow.log_param("train_loss_type", self.config.train_regression_loss._target_)
        mlflow.log_param("val_loss_type", self.config.val_regression_loss._target_)

    def init_parameters(self):
        self.train_max_age = self.config.dataset.train_max_age
        self.val_max_age = self.config.dataset.val_max_age
        self.train_min_quality = self.config.dataset.train_min_quality
        self.val_min_quality = self.config.dataset.val_min_quality
        self.train_faceview = self.config.dataset.train_faceview
        self.val_faceview = self.config.dataset.val_faceview

        self.max_days = DAYS_IN_YEAR * self.train_max_age
        self.days_scale = self.max_days if self.config.dataset.normalize_y else DAYS_IN_YEAR
        self.max_dob_error = self.config.dataset.max_dob_error
        self.dataset_metadata_path = os.path.join(
            self.config.dataset.basepath, self.config.dataset.metadata
        )
        self.dataset_images_path = os.path.join(
            self.config.dataset.basepath, self.config.dataset.images
        )
        self.sex = self.config.dataset.sex
        self.sex = None if self.sex == "" else self.sex

        self.learning_rate = self.config.training.learning_rate
        self.batch_size = self.config.training.batch_size
        self.prediction_batch_size = self.config.training.prediction_batch_size
        self.validation_batch_size = self.config.training.validation_batch_size
        self.epochs = self.config.training.epochs
        self.in_mem = self.config.dataset.in_memory
        self.resume = self.config.resume

        self.kfold = self.config.kfold

        self.early_stopping_patience = self.config.training.early_stopping_patience

        self.img_size = (self.config.dataset.im_size, self.config.dataset.im_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")

        self.name = self.config.name + f"_{self.kfold}folds"
        if self.config.kfold_index != -1:
            self.name += f"_k:{self.config.kfold_index}"

        print(self.name)

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
        if isinstance(x, dict):
            for key in x.keys():
                x[key] = x[key].to(device)
        elif isinstance(x, list):
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

    def multihead_train_step(self, x, y, backbone, heads, criterions, weights):
        backbone_features = backbone(x)
        losses = {}
        for head_name, head_model in heads.items():
            y_hat = head_model(backbone_features)
            if y[head_name].dtype == torch.float64:
                y[head_name] = y[head_name].type(torch.float)
            loss = criterions[head_name](y_hat, y[head_name])
            losses[head_name] = loss

        # Weight losses
        losses = {head_name: (value * weights[head_name]) for head_name, value in losses.items()}
        # Sum losses
        loss = torch.sum(torch.stack(list(losses.values())))

        losses["train"] = loss.float()

        return losses

    def basic_train_step(self, loader, backbone, heads, criterions, weights, device):
        data = next(iter(loader))
        x = data.pop("input")
        x, y = self.xy_to_device(x, data, device)

        if self.config.training.use_float16:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                losses = self.multihead_train_step(x, y, backbone, heads, criterions, weights)
        else:
            losses = self.multihead_train_step(x, y, backbone, heads, criterions, weights)

        return losses

    def test(self):
        raise ValueError("You must subclass self.test() method")

    def init(self):
        self.output_dir = os.path.join(self.base_output_dir, f"{self.train_index}")
        os.makedirs(self.output_dir, exist_ok=True)

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
        assert self.config.train or self.config.test, "At least train or test must be true"

        training_score = None
        test_score = None

        assert self.config.kfold > 1, "kfold must be set to k > 1"

        # If the kfold index is set to -1, loop over all possible k
        if self.config.kfold_index == -1:
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
            cross_val_scores_path = os.path.join(self.output_dir, "cross_val_scores.json")
            with open(cross_val_scores_path, "w") as f:
                import json

                f.write(json.dumps(test_scores))

            # Return mean score
            test_scores = np.mean(test_scores)
        else:
            self.train_index = self.config.kfold_index
            self.init()
            if self.config.train:
                training_score = self.train()
            if self.config.test:
                test_score = self.test()

        self.score = training_score
        if self.config.test:
            self.score = test_score

        log.info(f"Final score : {self.score}")

        mlflow.end_run()

        return self.score

    def setup_age_steps(self):
        # Age steps with number of epochs
        # This mean that we set the max age at time 0 to the global max age
        # It is the default behavior
        epoch_step = self.epochs
        age_step = self.train_max_age

        # We increase the max age of both train/val dataset incrementally
        # epoch_step = 10
        # age_step = 0.2

        age_steps = {
            i * epoch_step: min(self.train_max_age, (i + 1) * age_step)
            for i in range(int(self.train_max_age // age_step) + 1)
        }
        return age_steps

    def train_mode():
        raise NotImplementedError()

    def optimize(self, loss, scaler):
        if self.config.training.use_float16:
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

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
            df.to_csv(os.path.join(self.output_dir, f"val_raw_predictions_{self.train_index}.csv"))
        else:
            log.info(f"Val loss did not improved from {best_val:.4f}")
        return best_val, improved

    def early_stopping(self, improved):
        if not improved:
            self.improved_since += 1
        else:
            self.improved_since = 0
        if self.improved_since >= self.early_stopping_patience:
            log.info(
                f"Stopping training since validation loss did not improved for {self.early_stopping_patience} epochs"
            )
            return True
        return False

    def train(self):
        if self.resume:
            self.model = load(
                self.model,
                f"regression_{self.train_index}",
                exp_name=self.name,
                output_dir=self.output_dir,
            )

        age_steps = self.setup_age_steps()
        log.info(f"Using ages steps: {age_steps}")

        self.prof = None
        if self.config.profile:
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=12, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(self.output_dir, "profiler")
                ),
                record_shapes=False,
                with_stack=False,
            )
            self.prof.start()

        scaler = torch.cuda.amp.GradScaler()

        self.scheduler = None
        if self.config.training.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.config.training.learning_rate * 0.01,
                max_lr=self.config.training.learning_rate,
                cycle_momentum=False,
            )

        best_val = np.inf
        pbar = tqdm(range(self.epochs))
        self.improved_since = 0
        for epoch in pbar:
            train_loss = self.train_epoch(epoch, age_steps, scaler, pbar)
            epoch_val, val_df, val_losses = self.validate_epoch(epoch)
            best_val, improved = self.save_best_val_loss(epoch_val, best_val, val_df)

            mlflow.log_metric("best_val_loss", best_val, step=epoch)

            if self.early_stopping(improved):
                break

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

        if self.config.profile:
            self.prof.stop()
        return best_val

    def validate_epoch(self, epoch):
        raise NotImplementedError()

    def train_epoch(self, epoch, age_steps, scaler, pbar):
        self.update_from_ages_steps(age_steps, epoch)

        self.train_mode()

        # Train for one epoch
        train_losses = None
        for i in tqdm(range(len(self.train_loader)), leave=True):
            step_losses = self.train_step()
            train_loss = step_losses["train"]

            self.optimize(train_loss, scaler)

            if self.scheduler is not None:
                self.scheduler.step()

            # Convert losses to float to use for display
            losses = {
                loss_name: [loss_value.detach().cpu().numpy()]
                for loss_name, loss_value in step_losses.items()
            }
            if train_losses is None:
                train_losses = losses
            else:
                for loss_name, loss_value in losses.items():
                    train_losses[loss_name] += loss_value

            # Create training description
            train_description_str = [
                f"{loss_name}: {(np.mean(loss_values)):.5f}"
                for loss_name, loss_values in train_losses.items()
            ]
            train_description_str = " - ".join(train_description_str)
            pbar.set_description(train_description_str)

            if self.prof:
                self.prof.step()

        log.info(train_description_str)

        # Compute and log mean losses for this epoch
        mean_losses = {
            loss_name: np.mean(loss_values) for loss_name, loss_values in train_losses.items()
        }
        mlflow.log_metrics(mean_losses, step=epoch)

        return mean_losses["train"]
