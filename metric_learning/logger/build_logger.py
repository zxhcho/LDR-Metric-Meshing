from utils.registry import Register

LOGGER_REGISTRY = Register()


def build_logger(config, mode):
    logger_type = config.logger.type
    return LOGGER_REGISTRY[logger_type](config, mode)
