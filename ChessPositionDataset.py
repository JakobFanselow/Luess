import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class ChessPositionDataset(Dataset):
    def __init__(self,file_path, data_name = "positions", label_name = "labels", transform=None):
        self.file_path = file_path
        self.data_name = data_name
        self.label_name = label_name
        self.transform = transform

        with h5py.File(self.file_path, 'r') as f:
            self.dataset_len = len(f[self.data_name])
            
        self.file = None

    def __len__(self):
        return self.dataset_len

    def __getitem__(self,idx):
        if self.file == None:
            self.file = h5py.File(self.file_path,'r')
        
        data = self.file[self.data_name][idx]
        label = self.file[self.label_name][idx]

        shifts = torch.arange(63, -1, -1, dtype=torch.int64)
        data_tensor = torch.from_numpy(data.astype('int64'))
        unpacked = (data_tensor.unsqueeze(-1) >> shifts) & 1
        data = unpacked.float()
        data = data.view(15, 8, 8)



        label = torch.from_numpy(label).float()

        if self.transform:
            data = self.transform(data)
        
        return data, label