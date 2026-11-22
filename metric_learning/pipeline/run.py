import argparse
import yaml
from pipeline.trainer.build_trainer import build_trainer
from pipeline.infer.build_infer import build_infer
from pipeline.infer.infer import Infer
from config.config_yaml import Config


def run(config, mode):
    if mode == 'train':
        trainer = build_trainer(config)
        trainer.train(total_epoch=config.epoch.n_epoch)
    else:
        infer = build_infer(config=config, mode=mode)
        infer.infer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config file")
    parser.add_argument("--mode", type=str, help="mode", default="train")
    args = parser.parse_args()

    config_file = args.config
    mode = args.mode

    with open(config_file, 'r') as f:
        config_node = yaml.safe_load(f)
    config = Config(**config_node)

    run(config, mode)
