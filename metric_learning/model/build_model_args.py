from utils.registry import Register

MODEL_ARGS_REGISTRY = Register()

feature_channel = {
    "xyz": 3,
    "normal": 3,
    "color": 3,
    "basis": 6,
    "frame": 9,
    "mean_curv": 1,
    "pcd_curv": 1,
    "gauss_curv": 1,
    "curv_frame": 8,
    "hks": 16,
    "field": 1,
    "signal": 256,
}


def sum_feature_channel(feature_types: list):
    if len(feature_types) == 0:
        return 0
    return sum(feature_channel[feature_type] for feature_type in feature_types)


@MODEL_ARGS_REGISTRY("DiscreteEdgeMetricDiffusionNet")
def ConnectionNetArgs(config):
    channel = sum_feature_channel(config.model.feature_types)
    model_args = {
        "in_channels": channel,
    }
    return model_args
