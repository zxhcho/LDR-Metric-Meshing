from utils.registry import Register

VISUALIZER_REGISTRY = Register()


def build_visualizer(config):
    visualizer_type = config.infer.visualizer.type
    return VISUALIZER_REGISTRY[visualizer_type]
