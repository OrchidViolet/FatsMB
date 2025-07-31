
from abc import *


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self,
            dataset
            ):
        save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.train_b = dataset['train_b']
        self.val_b = dataset['val_b']
        self.val_num = dataset['val_num']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.bmap = dataset['bmap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        self.behavior_count = len(self.bmap)


    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_val_loader(self):
        pass