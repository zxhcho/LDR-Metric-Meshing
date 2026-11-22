from utils.registry import Register
from torch.utils.data import DataLoader
from dataset.build_dataset import build_dataset
from dataloader.collate_fn import dic_cat_collate_fn, dic_stack_collate_fn

DATALOADER_REGISTRY = Register()


@DATALOADER_REGISTRY("DiffusionNet")
def DiffusionNetDataloader(config, dataset, mode):
    shuffle = True if mode == "train" else False
    return DataLoader(dataset, batch_size=None, shuffle=shuffle)


@DATALOADER_REGISTRY("DicCat")
def PointCloudDataloader(config, dataset, mode):
    shuffle = True if mode == "train" else False
    batch_size = getattr(config.dataloader.batch_size, mode)
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dic_cat_collate_fn, pin_memory=True)


@DATALOADER_REGISTRY("DicStack")
def DGCNNDataloader(config, dataset, mode):
    shuffle = True if mode == "train" else False
    batch_size = getattr(config.dataloader.batch_size, mode)
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dic_stack_collate_fn, pin_memory=True)


DATALOADER_REGISTRY["PointCloud"] = DATALOADER_REGISTRY["DicCat"]
DATALOADER_REGISTRY["DGCNN"] = DATALOADER_REGISTRY["DicStack"]
DATALOADER_REGISTRY["PointNetPP"] = DATALOADER_REGISTRY["DicStack"]
DATALOADER_REGISTRY["PointNet"] = DATALOADER_REGISTRY["DicStack"]


def build_dataloader(config, dataset_type, mode, root_dir):
    dataset = build_dataset(config, dataset_type, mode, root_dir)
    dataloader_type = config.dataloader.type
    return DATALOADER_REGISTRY[dataloader_type](config, dataset, mode)
