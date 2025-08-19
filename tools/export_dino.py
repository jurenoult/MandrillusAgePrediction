import torch
import click
from mandrillage.models import DinoV2, RegressionHead

# Follow https://github.com/facebookresearch/dinov2/issues/19
# To update your dinov2 vision transformer


class xyz_model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tensor):
        ff = self.model(tensor)
        return ff


def convert_dino_model(model_path, output_path, dino_type="small"):
    # Load a default backbone model
    baseline_backbone_model = DinoV2(dino_type).to("cpu")

    # Load model
    model = torch.load(model_path).to("cpu")
    backbone_statedict = model.backbone.state_dict()    
    baseline_backbone_model.load_state_dict(backbone_statedict)
    model.backbone = baseline_backbone_model

    model.eval()
    input_data = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    output = model(input_data)
    print(f"Inference output: {output}")
    torch.onnx.export(model, input_data, output_path, input_names=["input"])


@click.command()
@click.option(
    "--model_path",
    required=True,
    help="Path to model to load.",
)
@click.option(
    "--export_path",
    required=True,
    help="Path to exported model.",
)
@click.option(
    "--dino_type",
    required=False,
    default="large",
    help="Dino model type (small,medium,large)",
)
def main(model_path, export_path, dino_type):
    convert_dino_model(model_path, export_path, dino_type)


if __name__ == "__main__":
    main()
