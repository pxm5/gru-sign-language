import torch
from torch.utils.data import Dataset
import utils
import pandas as pd


class TimeSeriesDataset(Dataset):
    def __init__(self, data:dict):
        df = pd.read_csv("Data/wlasl_class_list.txt", delimiter="\t", names=["Id", "Name"])
        formatted_data = utils.format_data(data)
        X = formatted_data
        self.y = []

        for key in data:
            self.y.append(df.loc[df['Name'] == key,'Id'].values[0])
        self.data = [[torch.Tensor(X[i]), torch.Tensor([self.y[i]] * len(X[i]))] for i in range(len(X))]
        
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def getLabel(self):
        return self.y