import mlflow
from .module import SegmentationLightningModule
from .dataset import DAVISDataModule
import pytorch_lightning as pl


def train():
    # Example usage:
    dm = DAVISDataModule("data/davis_annot_sam2")
    ckpt_path = "/home/rkarpins/projects/others/sam2/checkpoints/mandrill_checkpoint_v0.pt"
    model = SegmentationLightningModule(ckpt_path=ckpt_path, num_classes=1)
    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", precision="bf16-mixed")
    with mlflow.start_run():
        trainer.fit(model, dm)


if __name__ == "__main__":
    train()
