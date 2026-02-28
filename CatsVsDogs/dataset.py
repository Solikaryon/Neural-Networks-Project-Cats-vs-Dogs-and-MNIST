import os
import torch
import numpy as np
from torch.utils.data import Dataset

class PetImageDataset(Dataset):
    """
    Custom Dataset class to load .npy files from the pre-processing script.
    """
    def __init__(self, data_dir):
        """
        Constructor: Initializes the dataset.
        """
        self.data_dir = data_dir
        # Get a list of all .npy files
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        # Create labels based on the filename: 0 for 'Cat', 1 for 'Dog'
        # This assumes your pre-processor script creates files like "Cat_1.npy" or "Dog_1.npy"
        self.labels = [0 if f.startswith('Cat') else 1 for f in self.file_list]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Gets a single sample (image and label) from the dataset.
        Handles corrupt .npy files by loading a random different sample.
        """
        
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        try:
            img_array = np.load(file_path) 
            
            # Convert to PyTorch tensor
            # [H, W, C] -> [C, H, W]
            image_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            
            # Get the corresponding label
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            
            return image_tensor, label_tensor.unsqueeze(0)
        
        except EOFError:
            # If the data is corrupt then loads a random one
            print(f"\n[!] Warning: Corrupt file detected, skipping: {file_path}")
            new_idx = random.randint(0, len(self) - 1) 
            return self.__getitem__(new_idx)
        except Exception as e:
            print(f"\n[!] Error loading file {file_path}: {e}")
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)