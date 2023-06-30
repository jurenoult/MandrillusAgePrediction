import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.2")
def main(config: DictConfig):
    pipeline = hydra.utils.instantiate(config.pipeline)
    pipeline.set_config(config)
    score = pipeline.run()

    return score


if __name__ == "__main__":
    main()
