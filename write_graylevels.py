import os
import sys 
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, Tuple, Any, Dict

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

sys.path.append("../")
from modules import MinMaxNormalization, CropSquareResize

IM_SIZE = 224
BATCH_SIZE = 512
JUD_BINS = list(range(0, 257, 8))
NUM_WORKERS = 1 
torch.manual_seed(1120)
torch.set_num_threads(NUM_WORKERS * 2) # each cpu has 2 threads, so 30*8 = 240, and then we can run two processes on each cpu

class BUSDensityDataset(Dataset):
    """
    A custom dataset class for handling breast ultrasound density data.

    Args:
        annotations_file (str): Path to the CSV file containing annotations.
        len_file (int): The number of samples to select from the annotations file.
        img_dir (str): Directory with all the images.
        split_img_dir (Optional[str]): Directory with split scan images.
        img_size (int): Size to which images are resized (img_size x img_size).
        transform (Optional[callable], optional): A function/transform to apply to the images. Default is None.
        target_transform (Optional[callable], optional): A function/transform to apply to the labels. Default is None.
    """
    def __init__(self, 
                 annotations_file: str, 
                 len_file: int, 
                 img_dir: str, 
                 split_img_dir: Optional[str], 
                 img_size: int, 
                 transform: Optional[Any] = None, 
                 target_transform: Optional[Any] = None) -> None:
        
        n_sample = int(len_file)
        cols = ['Image_file', 'ANALYSIS_ID', 'DENSITY_MG']
            
        self.dataframe = pd.read_csv(annotations_file, low_memory=False).loc[:, cols].sample(n_sample, axis=0, ignore_index=True)
        self.img_dir = img_dir
        self.split_img_dir = split_img_dir
        self.transform = transform
        self.img_size = (img_size, img_size)
        self.annotations_file = annotations_file
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """
        Retrieves an item from the dataset at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple containing the image, label, patient ID, density, age (US), age (MG), BIRADS score, manufacturer, and cancer status.
        """
        path = self.dataframe.iloc[idx, 0]
        # in the paper data, split dual-view images are coded [image_id]_R/L.png after they are split
        # for the sample data, we don't provide a split_images path or any split images
        if (('_L.png' in path) or ('_R.png' in path)) and self.split_img_dir is not None:
            image = Image.open(os.path.join(self.split_img_dir, path)).convert('L')
        else:
            image = Image.open(os.path.join(self.img_dir, path)).convert('L')

        pat_id = torch.as_tensor(self.dataframe.iloc[idx, 1])
        density = torch.as_tensor(self.dataframe.iloc[idx, 2])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            density = self.target_transform(density)

        return image, pat_id, density

def get_dataloaders(batch_size, test_augs, create_numpy=False):
    '''
    Helper function to create the dataloaders with specified augmentations and batch size 
    
    Input: batch_size -> batch size, could be tuned (int)
           test_augs -> test/validation augmentations (torchvision.Compose)
           create_numpy -> whether or not to return numpy placeholders as well, T/F
           
    Returns: 4-tuple containing validation B and testing data loaders, and placeholder numpy arrays 
    
    '''
    
    validation_data_b = BUSDensityDataset(annotations_file='sample_data/sample_data.csv', 
                                          img_dir='sample_data/images', 
                                          split_img_dir=None,
                                          img_size=IM_SIZE, transform=test_augs, len_file=5)

    testing_data = BUSDensityDataset(annotations_file='sample_data/sample_data.csv', 
                                          img_dir='sample_data/images', 
                                          split_img_dir=None,
                                          img_size=IM_SIZE, transform=test_augs, len_file=5)
    
    val_b_loader = DataLoader(validation_data_b, batch_size=batch_size, num_workers=NUM_WORKERS)
    test_loader = DataLoader(testing_data, batch_size=batch_size, num_workers=NUM_WORKERS)

    # we store the graylevels in one numpy array for ease of use with sklearn 
    if create_numpy:
        val_b = dict()
        val_b_size = 5
        val_b['pat_ids'] = np.zeros(shape=(val_b_size, 1))
        val_b['gray_levels'] = np.zeros(shape=(val_b_size, 50176))
        val_b['density'] = np.zeros(shape=(val_b_size, 1))

        test = dict()
        test_size = 5
        test['pat_ids'] = np.zeros(shape=(test_size, 1))
        test['gray_levels'] = np.zeros(shape=(test_size, 50176))
        test['density'] = np.zeros(shape=(test_size, 1))

        return val_b_loader, test_loader, val_b, test
    
    return  val_b_loader, test_loader

def generate_histograms(dataloader: DataLoader, np_storage: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Generates predictions and stores them in a given numpy storage dictionary.

    Args:
        dataloader (DataLoader): DataLoader providing the batches of data.
        np_storage (Dict[str, np.ndarray]): Dictionary to store the predictions and additional data.

    Returns:
        Dict[str, np.ndarray]: Updated numpy storage dictionary with predictions and additional data.
    """
    start_index = 0
    
    for batch, (X,  pat, density) in enumerate(dataloader):
        length = len(density)
        np_storage['density'][start_index:start_index+length] = density.numpy().reshape(-1, 1)
        np_storage['pat_ids'][start_index:start_index+length] = pat.numpy().reshape(-1, 1)
        np_storage['gray_levels'][start_index:start_index+length] = X.numpy().reshape(-1, 50176)

        start_index += length

    return np_storage

if __name__ == '__main__':
    split_name = sys.argv[1] # should be one of valid or test
    norm_style = sys.argv[2] # should be one of norm or raw

    test_augs_norm = T.Compose([T.ToTensor(), MinMaxNormalization(), CropSquareResize()])
    test_augs = T.Compose([T.ToTensor(), CropSquareResize()])

    if norm_style == 'raw':
        val_b_loader, test_loader, val_b_np, test_np = get_dataloaders(512, test_augs, create_numpy=True)
        if split_name == 'valid':
            start_index = 0
            val_b_np = generate_histograms(val_b_loader, val_b_np)
            with open('graylevels_validation_b.pkl', 'wb') as file:
                pickle.dump(val_b_np, file)

        elif split_name == 'test':
            start_index = 0
            test_np = generate_histograms(test_loader, test_np)
            with open('graylevels_testing.pkl', 'wb') as file:
                pickle.dump(test_np, file)
    
    elif norm_style == 'norm':
        val_b_loader, test_loader, val_b_np, test_np = get_dataloaders(512, test_augs_norm, create_numpy=True)

        if split_name == 'valid':
            start_index = 0
            val_b_np = generate_histograms(val_b_loader, val_b_np)
            with open('graylevels_normed_validation_b.pkl', 'wb') as file:
                pickle.dump(val_b_np, file)

        elif split_name == 'test':
            start_index = 0
            test_np = generate_histograms(test_loader, test_np)
            with open('graylevels_normed_testing.pkl', 'wb') as file:
                pickle.dump(test_np, file)
