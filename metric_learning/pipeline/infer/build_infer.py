from utils.registry import Register


INFER_REGISTRY = Register()


def build_infer(config, mode):
    infer_type = config.infer.type
    return INFER_REGISTRY[infer_type](config, mode)
