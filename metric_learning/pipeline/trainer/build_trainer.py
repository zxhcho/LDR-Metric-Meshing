from utils.registry import Register

TRAINER_REGISTRY = Register()


def build_trainer(config):

    trainer_type = config.trainer.type

    return TRAINER_REGISTRY[trainer_type](config)
