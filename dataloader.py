import pickle
import os
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShapeDataset(Dataset):
    def __init__(self, pickle_dir):
        
        self.pickle_dir = pickle_dir
        
        # self.object_max_len = object_max_len
        
        onlyfiles = []
       
        onlyfiles = next(os.walk(pickle_dir), (None, None, []))[2]

        self.data_size = len(onlyfiles)
       
    def __len__(self):
        return self.data_size 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = str(int(idx))
        print(file)
        save_file_name = os.path.join(self.pickle_dir, file)     
        with open(save_file_name, 'rb') as outfile:
            self.data_items = pickle.load(outfile)
            self.data_items = self.data_items['desc_list']

       
        shape = torch.tensor(self.data_items["shape"])
        size = torch.tensor(self.data_items["size"])
        color = torch.tensor(self.data_items["color"])
        action = torch.tensor(self.data_items["action"])
        intent = torch.tensor(self.data_items["type"])

        hidden_state = torch.tensor(self.data_items['hidden_state'], device=device)
        
        last_hidden = torch.tensor(self.data_items['last_hidden'], device=device)
        return shape, size, color, action, intent, hidden_state, last_hidden

