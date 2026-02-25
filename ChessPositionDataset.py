import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ChessPositionDataset(Dataset):
    def __init__(self,file_path, data_name = "positions", label_name = "labels", transform=None):
        self.transform = transform

        with h5py.File(file_path, 'r') as f:
            self.data = torch.from_numpy(f[data_name][:]).share_memory_()
            self.labels = torch.from_numpy(f[label_name][:]).share_memory_()
            self.shifts = torch.arange(63, -1, -1, dtype=torch.int64)


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        
        data_bitmap = self.data[idx].to(torch.int64)


        unpacked = (data_bitmap.unsqueeze(-1) >> self.shifts) & 1
        data = unpacked.view(18, 8, 8).float()

        label = self.labels[idx].float()

        if self.transform:
            data = self.transform(data)

        return data, label