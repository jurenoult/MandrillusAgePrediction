import hydra
from omegaconf import DictConfig, OmegaConf

import mlflow.pytorch
from mlflow import MlflowClient


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def run(config: DictConfig):
    OmegaConf.resolve(config)
    pipeline = hydra.utils.instantiate(config.pipeline)

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]

    # Auto log all MLflow entities
    # mlflow.pytorch.autolog()

    pipeline.set_config(config, output_dir)

    # with mlflow.start_run() as run:
    score = pipeline.run()

    # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    return score


@hydra.main(config_path="config/", config_name="train.yaml", version_base="1.2")
def main(config: DictConfig):
    return run(config)


if __name__ == "__main__":
    main()
