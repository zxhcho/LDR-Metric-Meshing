from torch.utils.data import Dataset
from abc import abstractmethod, ABC


class BaseDataset(Dataset, ABC):
    def __init__(self, root_dir, mode) -> None:
        super().__init__()
        self.mode = mode  # "train", "evaluate", "predict"
        self.root_dir = root_dir

    @abstractmethod
    def load_filenames(self):
        raise NotImplementedError

    @abstractmethod
    def load_file(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
