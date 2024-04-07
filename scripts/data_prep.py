# modifying class to have only one band in label
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import rasterio as rio


class SiamDW_DataClass(Dataset):
    def __init__(self, data_path, metadata, set, learn_type, process_level, transform=None):
    # in metadata dataframe, select patches with set as train 
        self.data_path = data_path
        self.metadata = metadata
        self.set = set
        self.transform = transform
        self.process_level = process_level
        self.learn_type = learn_type

    def __len__(self):
        return len(self.metadata[(self.metadata['set']==self.set) & (self.metadata['learn_type']==self.learn_type)])

    def __getitem__(self, idx):
        # return tensor pair feature,label
        # read image
        metadata_set = self.metadata[(self.metadata['set']==self.set) & (self.metadata['learn_type']==self.learn_type)]
        feat_fname= (f"{self.process_level}_"
                      f"{self.learn_type}_"
                      f"{metadata_set.iloc[idx, 0]}_"
                      f"feats.tif")
        label_fname= (f"{self.process_level}_"
                      f"{self.learn_type}_"
                      f"{metadata_set.iloc[idx, 0]}_"
                      f"label.tif")

        feat_p = os.path.join(self.data_path, feat_fname)
        label_p = os.path.join(self.data_path, label_fname)
        
        with rio.open(feat_p) as src:
            feat = src.read().astype(np.int16)
        with rio.open(label_p) as src:
            label = src.read().astype(np.int16)

        # handle dtypes for tensor
        feat = torch.from_numpy(feat).to(torch.float32)
        label = torch.from_numpy(label).to(torch.float32)
        # if self.transform:
        #     feat = self.transform(feat)
            # label = self.transform(label)
        return feat, label[5,:,:].long(), metadata_set.iloc[idx, 0]
    

def prepare_loaders(input_dir, process_level, learn_type, batch_size):

    metadata = pd.read_csv(os.path.join(input_dir, 'meta_patches.csv'))
    train_data_path = os.path.join(input_dir, process_level, 'train')
    val_data_path = os.path.join(input_dir, process_level, 'test')
    
    train_dataset = SiamDW_DataClass(data_path=train_data_path, metadata= metadata, 
                                     set ='train', learn_type = learn_type, process_level = process_level)
    val_dataset = SiamDW_DataClass(data_path = val_data_path, metadata = metadata, 
                                   set = 'test', learn_type=learn_type, process_level=process_level)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    return train_loader, val_loader