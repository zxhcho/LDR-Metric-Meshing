"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""


import torch


def list_collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def dic_cat_collate_fn(batch):
    # Get the keys from the first dictionary in the batch
    keys = batch[0].keys()

    data = {key: [] for key in keys}  # Create an empty list for each key

    for item in batch:
        for key in keys:
            # Append the value corresponding to each key
            data[key].append(item[key])

    offset, count = [], 0
    for item in data['feat']:
        count += item.shape[0]
        offset.append(count)

    data_rtn = {key: torch.cat(value) for key, value in data.items()}
    data_rtn['offset'] = torch.IntTensor(offset)

    return data_rtn


def dic_stack_collate_fn(batch):
    # Get the keys from the first dictionary in the batch
    keys = batch[0].keys()

    data = {key: [] for key in keys}  # Create an empty list for each key

    for item in batch:
        for key in keys:
            # Append the value corresponding to each key
            data[key].append(item[key])

    stacked_data = {key: torch.stack(value) for key, value in data.items()}

    return stacked_data
