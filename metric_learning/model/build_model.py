from model.build_model_args import MODEL_ARGS_REGISTRY


def find_model_and_import(model_type):
    if model_type == "DiscreteEdgeMetricDiffusionNet":
        from model.MetricNet.metric_diffusion_net import DiscreteEdgeMetricDiffusionNet
        return DiscreteEdgeMetricDiffusionNet
    elif model_type is None:
        return None
    raise KeyError(model_type)


def build_model(model_type, config):
    model_class = find_model_and_import(model_type)
    if model_class is None:
        return None
    model_args = MODEL_ARGS_REGISTRY[model_type](config)
    return model_class(**model_args, config=config)
