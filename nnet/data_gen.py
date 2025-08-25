import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CoverTypeDataset(Dataset):

    def __init__(self, file_path: str):
        self.dataframe = pd.read_csv(file_path)


    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
         
        features = torch.tensor(self.dataframe.iloc[idx, :-1].values, dtype = torch.float32)
        labels = torch.tensor(self.dataframe.iloc[idx, -1], dtype = torch.long)

        return features, labels