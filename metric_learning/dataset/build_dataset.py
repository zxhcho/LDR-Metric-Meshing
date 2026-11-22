from utils.registry import Register

DATASET_REGISTRY = Register()
DATASET_FILENAMES_REGISTRY = Register()


def build_dataset(config, dataset_type, mode, root_dir):
    return DATASET_REGISTRY[dataset_type](config=config, root_dir=root_dir, mode=mode)
