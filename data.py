import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from glob import glob

# Dataset Class adapted from
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# and https://github.com/mvsjober/pytorch-hdf5/blob/master/pytorch_dvc_cnn.py
class XrayDataSet(Dataset):
    """
    Custom Dataset for Xray images from Dr. Albert Hsiao
    """
    def __init__(self, annotations_df: pd.DataFrame, fn_stem: str, transform=None):
        self.img_labels = annotations_df
        self.fn_stem = fn_stem
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        key = self.img_labels.iloc[idx, 0]
        hdf5_fnum = self.img_labels.iloc[idx,-1]
        hdf5_path = self.fn_stem + str(hdf5_fnum) + '.hdf5'
        with h5py.File(hdf5_path, 'r') as hdf5:
            image = np.array(hdf5[key])
            # log scale
            image = np.log(1 + image)
            # Scale the dynamic range to 0-255
            image = (image - image.min()) * 255 / (image.max() - image.min())
            # Make datatype 8 bits [0-255] 
            image = ToTensor()(image.astype(np.uint8))
            # Create Fake RGB Image
            image = image.repeat(3,1,1)
            
            if self.transform:
                image = self.transform(image)
        # Take Raw BNPP values
        value = self.img_labels.iloc[idx, 5]
        value = torch.tensor(np.array(value).astype(np.float32)) + 1
        value = torch.log(value).reshape(1)
        return image, value


def get_key_to_file_dict(config):
    """
    Creates a pandas Series that maps with index
    as the unique image key and value as file number
    
    Returns k_to_f_series (Pandas.Series): unique image key and value as file number
    """
    keys = []
    fids = []
    hdf5_fs = glob(config['filepaths']['data_dir_path'] + '*.hdf5') # TODO: Update File Path
    for filename in hdf5_fs:
        f = h5py.File(filename, 'r')
        fid = filename.split('1024_')[1].split('.')[0]
        f_keys = f.keys()
        keys = keys + list(f_keys)
        fids = fids + [fid] * len(f_keys)
    k_to_f = dict(zip(keys, fids))
    k_to_f_series = pd.Series(k_to_f, name='k2f')
    return k_to_f_series

def get_dataset(df, fn_stem: str, transform=None):
    return XrayDataSet(df, fn_stem, transform=transform)

def merge_df_k2f(fn: str, k2f: pd.Series):
    """
    Matches the keys in the dataframe to the file number
    
    Parameters:
    fn (str): filename
    k2f (pd.Series): key to file number series
    
    Returns:
    df (pd.DataFrame): Dataframe with image key, BNPP value, and file number
    """
    df = pd.read_csv(fn)
    df = df.merge(k2f, left_on='unique_key', right_index=True)
    return df
        
def get_dataloaders(config, train_transform=None, test_transform=None):
    """
    Creates Pytorch DataLoaders for Xray Dataset
    
    Parameters:
    train_fn (str): training csv file name
    val_fn (str): validation csv file name
    test_fn (str): test csv file name
    bz (int): batch size
    shuffle (bool): Shuffle the data
    num_workers (int): The number of dataloader workers to use
    train_transform: Image transformations for the training dataset
    test_transform: A torch transform create with compose or a single transform
    
    Returns:
    train_dataloader (DataLoader): training set dataloader
    val_dataloader (DataLoader): validation set dataloader
    test_dataloader (DataLoader): test set dataloader
    """
    bs = config['dataloaders']['batch_size']
    shuffle = config['dataloaders']['batch_size']
    num_workers = config['dataloaders']['num_workers']
    train_fn = config['filepaths']['train_dataset']
    val_fn = config['filepaths']['val_dataset']
    test_fn = config['filepaths']['test_dataset']
    
    k2f_dict = get_key_to_file_dict(config)
    train_df = merge_df_k2f(train_fn, k2f_dict)
    val_df = merge_df_k2f(val_fn, k2f_dict)
    test_df = merge_df_k2f(test_fn, k2f_dict)
    
    fn_stem = os.path.join(config['filepaths']['data_dir_path'], config['filepaths']['hdf5_stem']) 
    train_dataset = get_dataset(train_df, fn_stem, transform=train_transform)
    val_dataset = get_dataset(val_df, fn_stem, transform=train_transform)
    test_dataset = get_dataset(test_df, fn_stem, transform=test_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
    
    return train_dataloader, val_dataloader, test_dataloader
                           
