# modifying class to have only one band in label
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import rasterio as rio
from torchvision import transforms
import csv
from sklearn.model_selection import StratifiedKFold

class SiamDW_DataClass(Dataset):
    def __init__(self, data_path, metadata, set, learn_type, process_level, input_type, lis_patch_ids = None, feat_transform=None, label_transform = None):
    # in metadata dataframe, select patches with set as train 
        self.data_path = data_path
        self.set = set
        self.process_level = process_level
        self.learn_type = learn_type
        self.input_type = input_type
        self.metadata = metadata
        self.feat_transform = feat_transform
        self.label_transform = label_transform

        # subset based on set (train/test) and learn_type (csl/ssl)
        self.metadata_set = self.metadata[(self.metadata['set']==self.set) & (self.metadata['learn_type']==self.learn_type)]
        
        # subset again if list of patch_ids is given - this will be used during training for implementing stratified kfold
        if lis_patch_ids is not None:
            self.metadata_set = self.metadata_set[self.metadata_set['patch_id'].isin(lis_patch_ids)]

    def __len__(self):
        return len(self.metadata_set)

    def __getitem__(self, idx):

        # Load label
        label_fname= (f"{self.process_level}_"
                f"{self.learn_type}_"
                f"{self.metadata_set.iloc[idx, 0]}_"
                f"label.tif")
        label_p = os.path.join(self.data_path, label_fname)
        with rio.open(label_p) as src:
            label = src.read().astype(np.int16)

        # Load feature (based on input type)
        if self.input_type == 's2':                 # multipsectral input
            feat_fname= (f"{self.process_level}_"
                        f"{self.learn_type}_"
                        f"{self.metadata_set.iloc[idx, 0]}_"
                        f"feats.tif")
            feat_p = os.path.join(self.data_path, feat_fname)
            with rio.open(feat_p) as src:
                feat = src.read().astype(np.int16)
       
        elif 'siam' in self.input_type:             # siam input
            feat = self.convert_to_rgb(label, self.input_type)

        # Convert to torch tensors of type float32
        feat = torch.from_numpy(feat).to(torch.float32)
        label = torch.from_numpy(label).to(torch.float32)

        # move up 255 to a value after max of the granularity, subtract 1 from last 4 bands i.e. siam bands of label to make it 0-indexed
        if self.learn_type == 'ssl':
            #set 255 to value after max value in the band
            label[1]=torch.where(label[1]==255, 19, label[1])
            label[2]=torch.where(label[2]==255, 34, label[2])
            label[3]=torch.where(label[3]==255, 49, label[3])
            label[4]=torch.where(label[4]==255, 97, label[4])
            label[1:] = label[1:] - 1
        
        if self.feat_transform is not None: 
            feat = self.feat_transform(feat)     
        if self.label_transform is not None:
            label = self.label_transform(label)    # for padding 

        return feat, label.long(), self.metadata_set.iloc[idx, 0] #get the patch id to save results
    
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
    
    def convert_to_rgb(self, label, input_type):
        #path to legend csv
        file_name = 'lgd_' + input_type + '.csv'
        legend_path = os.path.split(os.path.split(self.data_path)[0])[0] # remove last two levels of path to get to the data folder
        legend_path = os.path.join(legend_path, 'legends',file_name)
        rgb_dict = self.read_legend(legend_path) # this returns a dictionary with class values as keys and rgb values as tuples
        rgb_image = np.zeros((3, label.shape[1], label.shape[2]), dtype=np.uint8)

        # band 0 is scl, band 1 in label is siam_18, band 2 is siam_33, band 3 is siam_48, band 4 is siam_96
        if input_type == 'siam_18':
            band = 1
        elif input_type == 'siam_33':
            band = 2
        elif input_type == 'siam_48':
            band = 3
        elif input_type == 'siam_96':
            band = 4
        
        # Create RGB image by assigning values to the image based on the tensor values of the siam band of interest
        for value, rgb in rgb_dict.items():
            rgb_image[0, label[band] == value] = rgb[0]
            rgb_image[1, label[band] == value] = rgb[1]
            rgb_image[2, label[band] == value] = rgb[2]

        return rgb_image

class NormalizeImage:

    def __init__(self, input_type):
        super(NormalizeImage, self).__init__()
        self.input_type = input_type
        
    def __call__(self, image):
        if self.input_type == 's2':
            return self.normalize_s2(image)
        elif 'siam' in self.input_type:
            return self.normalize_siam(image)
        
    def normalize_s2(self, image):
        # Normalize image to the 1st and 99th percentile
        num_bands, _, _ = image.shape
        flattened_image = image.view(num_bands, -1)

        # Calculate the 1st and 99th percentile along the band dimension
        min_percentiles = torch.kthvalue(flattened_image, int(flattened_image.size(1) * 0.01), dim=1).values
        max_percentiles = torch.kthvalue(flattened_image, int(flattened_image.size(1) * 0.99), dim=1).values

        # Reshape percentiles to have shape (num_bands, 1, 1). This is necessary for element-wise operations
        min_percentiles = min_percentiles[:, None, None]
        max_percentiles = max_percentiles[:, None, None]

        # Clip image values to the range defined by percentiles
        normalized_image = torch.clamp((image - min_percentiles) / (max_percentiles - min_percentiles), 0, 1)
        return normalized_image
    
    def normalize_siam(self, image):
        # divide all three bands by 255
        normalized_image = image/255.0
        return normalized_image

class Padding:
    def __init__(self, output_size=(512, 512)):
        self.output_size = output_size

    def __call__(self, image):
        _, h, w = image.shape
        new_h, new_w = self.output_size

        top_pad = (new_h - h) // 2
        bottom_pad = new_h - h - top_pad
        left_pad = (new_w - w) // 2
        right_pad = new_w - w - left_pad

        padded_image = torch.nn.functional.pad(image, (left_pad, right_pad, top_pad, bottom_pad))
        return padded_image

def prepare_trainval_loaders(input_dir, process_level, learn_type, input_type, batch_size,train_pids,val_pids):
    metadata = pd.read_csv(os.path.join(input_dir, 'meta_patches.csv'))
    train_data_path = os.path.join(input_dir, process_level, 'train')

    feat_transform = transforms.Compose([NormalizeImage(input_type=input_type), Padding((512, 512))])
    label_transform = transforms.Compose([Padding((512, 512))])

    args={'data_path':train_data_path, 'metadata':metadata, 'set':'train', 'learn_type':learn_type, 'process_level':process_level, 
          'input_type':input_type, 'feat_transform':feat_transform, 'label_transform':label_transform}
    
    train_dataset = SiamDW_DataClass(lis_patch_ids=train_pids, **args)
    val_dataset = SiamDW_DataClass(lis_patch_ids=val_pids, **args)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    return train_loader, val_loader

def generate_stratified_folds(input_dir, process_level, learn_type, input_type, batch_size, n_splits=3):
    # look through input_dir and look for test_files_{i}.pkl files

    for i in range(n_splits):
        train_pids = pd.read_pickle(os.path.join(input_dir, f'{learn_type}_train_files_{i}.pkl'))
        val_pids = pd.read_pickle(os.path.join(input_dir, f'{learn_type}_val_files_{i}.pkl'))
        print(f'Fold {i} - Train: {len(train_pids)}, Val: {len(val_pids)}')
        train_loader, val_loader = prepare_trainval_loaders(input_dir, process_level, learn_type, input_type, batch_size,train_pids,val_pids)
        yield train_loader, val_loader

def prepare_test_loader(input_dir, process_level, learn_type, input_type, batch_size):
    metadata = pd.read_csv(os.path.join(input_dir, 'meta_patches.csv'))
    test_data_path = os.path.join(input_dir, process_level, 'test')

    feat_transform = transforms.Compose([NormalizeImage(input_type=input_type), Padding((512, 512))])
    label_transform = transforms.Compose([Padding((512, 512))])

    args={'data_path':test_data_path, 'metadata':metadata, 'set':'test', 'learn_type':learn_type, 'process_level':process_level, 
          'input_type':input_type, 'feat_transform':feat_transform, 'label_transform':label_transform}
    
    test_dataset = SiamDW_DataClass(**args)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    return test_loader
