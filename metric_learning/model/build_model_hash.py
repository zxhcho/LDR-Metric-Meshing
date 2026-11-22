from utils.registry import Register

MODEL_HASH_REGISTRY = Register()


@MODEL_HASH_REGISTRY("DiscreteEdgeMetricDiffusionNet")
def build_ff_hash(config):
    model_prefix = "DiscreteEdge_{}_{}_{}_Cri_{}_{}_Epo_{}_Opt_{}_{}_Sche_{}_{}".format(
        config.model.diffusion_net.N_block,
        config.model.diffusion_net.C_width,
        "_".join(config.model.feature_types),

        "_".join(config.criterion.types),
        "_".join([str(weight) for weight in config.criterion.weights]),

        config.epoch.n_epoch,

        config.optimizer.type,
        config.optimizer.lr,

        config.scheduler.step_size,
        config.scheduler.gamma)

    return model_prefix


def build_model_hash(config):
    model_type = config.model.type
    hash_id = MODEL_HASH_REGISTRY[model_type](config)
    if len(config.comment) != 0:
        hash_id = f"{config.comment}_{hash_id}"
    return hash_id
