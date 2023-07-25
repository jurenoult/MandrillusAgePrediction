import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.2")
def main(config: DictConfig):
    pipeline = hydra.utils.instantiate(config.pipeline)

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]

    pipeline.set_config(config, output_dir)
    score = pipeline.run()

    return score


if __name__ == "__main__":
    main()
