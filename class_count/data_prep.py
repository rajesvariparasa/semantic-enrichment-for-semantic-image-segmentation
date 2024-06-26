# modifying class to have only one band in label
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import rasterio as rio
from torchvision import transforms
import csv

class SiamDW_DataClass(Dataset):
    def __init__(self, data_path, metadata, set, learn_type, process_level, input_type, transform=None):
    # in metadata dataframe, select patches with set as train 
        self.data_path = data_path
        self.metadata = metadata
        self.set = set
        self.transform = transform
        self.process_level = process_level
        self.learn_type = learn_type
        self.input_type = input_type

    def __len__(self):
        return len(self.metadata[(self.metadata['set']==self.set) & (self.metadata['learn_type']==self.learn_type)])

    def __getitem__(self, idx):
        metadata_set = self.metadata[(self.metadata['set']==self.set) & (self.metadata['learn_type']==self.learn_type)]
        
        # Load label
        label_fname= (f"{self.process_level}_"
                f"{self.learn_type}_"
                f"{metadata_set.iloc[idx, 0]}_"
                f"label.tif")
        
        label_p = os.path.join(self.data_path, label_fname)
        with rio.open(label_p) as src:
            label = src.read().astype(np.int16)

        # Load feature if input_type is s2
        if self.input_type == 's2':
            feat_fname= (f"{self.process_level}_"
                        f"{self.learn_type}_"
                        f"{metadata_set.iloc[idx, 0]}_"
                        f"feats.tif")

            feat_p = os.path.join(self.data_path, feat_fname)
            with rio.open(feat_p) as src:
                feat = src.read().astype(np.int16)

        # else set siam from label file as feature
        elif 'siam' in self.input_type:
            feat = self.convert_to_rgb(label, self.input_type)

        feat = torch.from_numpy(feat).to(torch.float32)
        label = torch.from_numpy(label).to(torch.float32)
        if self.transform: # same transform for both siam and s2 currently (normalise based on bands in patch)
            feat = self.transform(feat)         

        return feat, label.long(), metadata_set.iloc[idx, 0]

    def convert_to_rgb(self, label, input_type):
        #path to legend csv
        file_name = 'lgd_' + input_type + '.csv'
        legend_path = os.path.split(os.path.split(self.data_path)[0])[0] # remove last two levels of path to get to the data folder
        legend_path = os.path.join(legend_path, 'legends',file_name)
        rgb_dict = self.read_legend(legend_path)
        rgb_image = np.zeros((3, label.shape[1], label.shape[2]), dtype=np.uint8)
        # band 2 in label is siam_18, band 3 is siam_33, band 4 is siam_48, band 5 is siam_96
        if input_type == 'siam_18':
            band = 2
        elif input_type == 'siam_33':
            band = 3
        elif input_type == 'siam_48':
            band = 4
        elif input_type == 'siam_96':
            band = 5
        for value, rgb in rgb_dict.items():
            rgb_image[0, label[band] == value] = rgb[0]
            rgb_image[1, label[band] == value] = rgb[1]
            rgb_image[2, label[band] == value] = rgb[2]
        return rgb_image

    def read_legend(self, legend_path):
        with open(legend_path, mode='r') as file:
            reader = csv.DictReader(file, delimiter=',')  
            rgb_dict = {}
            for row in reader:
                #print(row)
                value = int(row['value'])
                rgb = tuple(map(int, row['rgb'][1:-1].split(',')))  # Convert "(r, g, b)" string to tuple
                rgb_dict[value] = rgb
        return rgb_dict

class NormalizeImage:
    # def __init__(self):
    #     super(NormalizeImage, self).__init__()

    def __call__(self, image):
        num_bands, _, _ = image.shape
        flattened_image = image.view(num_bands, -1)

        # Calculate the 1st and 99th percentile along the band dimension
        min_percentiles = torch.kthvalue(flattened_image, int(flattened_image.size(1) * 0.01), dim=1).values
        max_percentiles = torch.kthvalue(flattened_image, int(flattened_image.size(1) * 0.99), dim=1).values

        # Reshape percentiles to have shape (num_bands, 1, 1)
        min_percentiles = min_percentiles[:, None, None]
        max_percentiles = max_percentiles[:, None, None]

        # Clip image values to the range defined by percentiles
        normalized_image = torch.clamp((image - min_percentiles) / (max_percentiles - min_percentiles), 0, 1)
        return normalized_image

def prepare_loaders(input_dir, process_level, learn_type, input_type, batch_size):

    metadata = pd.read_csv(os.path.join(input_dir, 'meta_patches.csv'))
    train_data_path = os.path.join(input_dir, process_level, 'train')
    val_data_path = os.path.join(input_dir, process_level, 'test')
    data_transform = transforms.Compose([NormalizeImage()])

    train_dataset = SiamDW_DataClass(data_path=train_data_path, metadata= metadata,transform = None,
                                     set ='train', learn_type = learn_type, process_level = process_level, input_type = input_type)
    val_dataset = SiamDW_DataClass(data_path = val_data_path, metadata = metadata, transform= None,
                                   set = 'test', learn_type=learn_type, process_level=process_level, input_type=input_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    return train_loader, val_loader