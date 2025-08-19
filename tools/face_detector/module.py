import mlflow
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn as nn
from torch import argmax
import torch

from .hiera_unet import SAM2UNet


class SegmentationLightningModule(pl.LightningModule):
    def __init__(self, ckpt_path, num_classes=2, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = SAM2UNet(checkpoint_path=ckpt_path)
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.n_logged = 0

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss(logits, masks)
        preds = argmax(logits, dim=1)
        acc = (preds == masks).float().mean()
        return loss, acc, imgs, masks, preds, logits

    def training_step(self, batch, batch_idx):
        loss, acc, imgs, masks, preds, logits = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_start(self):
        self.n_logged = 0

    def validation_step(self, batch, batch_idx):
        loss, acc, imgs, masks, preds, logits = self.step(batch)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)

        # Log visualization
        for i in range(imgs.shape[0]):
            self.log_segmentation_viz(imgs[i], masks[i], preds[i], logits[i], index=self.n_logged)
            self.n_logged += 1

        return {"val_loss": loss, "val_acc": acc}

    def log_segmentation_viz(self, img, mask, pred, logits, index=0):
        """Log visualization of input, GT mask, and prediction to MLflow"""
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        mask_np = mask.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        logits = logits.squeeze().detach().cpu().numpy()

        fig, axs = plt.subplots(1, 5, figsize=(12, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Input")
        axs[1].imshow(mask_np, cmap="jet", alpha=0.7)
        axs[1].set_title("GT Mask")
        axs[2].imshow(pred_np, cmap="jet", alpha=0.7)
        axs[2].set_title("Prediction")
        axs[3].imshow(logits[1], cmap="jet", alpha=0.7)
        axs[3].set_title("Logits")
        axs[4].imshow(img_np)
        axs[4].imshow(pred_np, cmap="jet", alpha=0.6)
        axs[4].set_title("Overlap")

        for ax in axs:
            ax.axis("off")

        mlflow.log_figure(fig, f"segmentation_{index}.png")
        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
