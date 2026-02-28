import os
import torch
import numpy as np
from torch.utils.data import Dataset

class MNISTNpyDataset(Dataset):
    """Dataset para cargar MNIST ya convertido a archivos .npy.
    Cada archivo se nombra <label>_<idx>.npy donde label es 0..9.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        # Etiqueta es la parte antes del primer '_'
        self.labels = [int(f.split('_')[0]) for f in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        fpath = os.path.join(self.data_dir, fname)
        try:
            arr = np.load(fpath)
            # arr shape: (28,28,1)
            tensor = torch.from_numpy(arr).permute(2,0,1)  # -> (1,28,28)
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.long)
            return tensor, label_tensor
        except Exception as e:
            # fallback simple: otro índice aleatorio
            import random
            new_idx = random.randint(0, len(self.file_list)-1)
            return self.__getitem__(new_idx)
