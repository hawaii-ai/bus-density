import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Any, List

import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from modules import MinMaxNormalization, CropSquareResize, CustomNet

IM_SIZE = 224
EPOCHS = 1000
NUM_WORKERS = 1 

torch.set_num_threads(NUM_WORKERS*2) 
torch.manual_seed(1120)
torch.set_float32_matmul_precision("medium")

class BUSDensityDataset(Dataset):
    def __init__(self, annotations_file, len_file, img_dir, split_img_dir, img_size, transform=None, target_transform=None):
        """
        Args:
            annotations_file (str): Path to the CSV file containing annotations (image paths and density labels).
            len_file (int): Number of samples to be loaded from the annotations file. Useful for debugging on a subset. 
            img_dir (str): Directory with all the original/cleaned images, depending on the data split. 
            split_img_dir (Optional[str]): Directory containing dual-view scans which were split into 2 examples, if any.
            img_size (Optional[int]): Desired size for the images for inputting to the network. 
            transform (Optional[Callable]): Optional torch transform(s) to be applied on an image.
            target_transform (Optional[Callable]): Optional torch transform(s) to be applied on a label.
        """
        n_sample = int(len_file)
        self.dataframe = pd.read_csv(annotations_file, low_memory=False).loc[:, ['Image_file', 'DENSITY_MG', 'ANALYSIS_ID', '_AGE_MG', '_AGE_US']].sample(n_sample, axis=0, ignore_index=True)
        self.img_dir = img_dir
        self.split_img_dir = split_img_dir
        self.transform = transform
        self.img_size = (img_size, img_size)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        path = self.dataframe.iloc[idx, 0]
        # in the paper data, split dual-view images are coded [image_id]_R/L.png after they are split
        # for the sample data, we don't provide a split_images path or any split images
        if (('_L.png' in path) or ('_R.png' in path)) and self.split_img_dir is not None:
            image =  Image.open(os.path.join(self.split_img_dir, path)).convert('L')
        else:
            image =  Image.open(os.path.join(self.img_dir, path)).convert('L')

        # in the paper data, need to -1 from the clinical density as it's coded 1-4 when we want 0-3
        label = torch.as_tensor(self.dataframe.iloc[idx, 1]-1)

        pat_id = torch.as_tensor(self.dataframe.iloc[idx, 2])
        age_us = torch.as_tensor(self.dataframe.iloc[idx, 4])
        age_mg = torch.as_tensor(self.dataframe.iloc[idx, 3])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, pat_id,  age_us, age_mg

def get_dataloaders(batch_size, test_augs, create_numpy=False, archs=None):
    '''
    Helper function to create the dataloaders with specified augmentations and batch size 
    
    Input: batch_size -> batch size, could be tuned (int)
           test_augs -> test/validation augmentations (torchvision.Compose)
           create_numpy -> whether or not to return numpy placeholders as well, T/F
           archs -> list of architectures to store preds for
           
    Returns: 3-tuple containing training, validation A, and validation B data loaders
    
    '''
    
    validation_data_a = BUSDensityDataset(annotations_file='sample_data/sample_data.csv', 
                                          img_dir='sample_data/images', 
                                          split_img_dir=None,
                                          img_size=IM_SIZE, transform=test_augs, len_file=5)
    
    validation_data_b = BUSDensityDataset(annotations_file='sample_data/sample_data.csv', 
                                          img_dir='sample_data/images', 
                                          split_img_dir=None,
                                          img_size=IM_SIZE, transform=test_augs, len_file=5)

    testing_data = BUSDensityDataset(annotations_file='sample_data/sample_data.csv', 
                                     img_dir='sample_data/images', 
                                     split_img_dir=None,
                                     img_size=IM_SIZE, transform=test_augs, len_file=5)
    
    val_a_loader = DataLoader(validation_data_a, batch_size=batch_size, num_workers=NUM_WORKERS)
    val_b_loader = DataLoader(validation_data_b, batch_size=batch_size, num_workers=NUM_WORKERS)
    test_loader = DataLoader(testing_data, batch_size=batch_size, num_workers=NUM_WORKERS)
    
    if create_numpy:
        val_a = dict()
        val_a['labels'] = np.zeros(shape=(5, 1))
        val_a['pat_ids'] = np.zeros(shape=(5, 1))
        val_a['age_us'] = np.zeros(shape=(5, 1))
        val_a['age_mg'] = np.zeros(shape=(5, 1))
        val_a['cancer'] = np.zeros(shape=(5, 1))

        val_b = dict()
        val_b['labels'] = np.zeros(shape=(5, 1))
        val_b['pat_ids'] = np.zeros(shape=(5, 1))
        val_b['age_us'] = np.zeros(shape=(5, 1))
        val_b['age_mg'] = np.zeros(shape=(5, 1))
        val_b['cancer'] = np.zeros(shape=(5, 1))

        test = dict()
        test['labels'] = np.zeros(shape=(5, 1))
        test['pat_ids'] = np.zeros(shape=(5, 1))
        test['age_us'] = np.zeros(shape=(5, 1))
        test['age_mg'] = np.zeros(shape=(5, 1))
        test['cancer'] = np.zeros(shape=(5, 1))

        if archs is not None:
            for name in archs:
                test[name+'_preds'] = np.zeros(shape=(5, 4))
                val_a[name+'_preds'] = np.zeros(shape=(5, 4))
                val_b[name+'_preds'] = np.zeros(shape=(5, 4))
        
        else:
            test['preds'] = np.zeros(shape=(5, 4))
            val_a['preds'] = np.zeros(shape=(5, 4))
            val_b['preds'] = np.zeros(shape=(5, 4))

        return val_a_loader, val_b_loader, test_loader, val_a, val_b, test
    
    return  val_a_loader, val_b_loader, test_loader
    
def generate_predictions(dataloader: DataLoader, 
                         np_storage: Dict[str, np.ndarray], 
                         softmax_obj: Any, 
                         archs: List[Module], 
                         arch_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Generates predictions using multiple models and stores them in a given numpy storage dictionary.

    Args:
        dataloader (DataLoader): DataLoader providing the batches of data.
        np_storage (Dict[str, np.ndarray]): Dictionary to store the predictions and additional data.
        softmax_obj (Any): Softmax function or object to apply to the model outputs.
        archs (List[Module]): List of models to use for generating predictions.
        arch_names (List[str]): List of model names corresponding to the models in archs.

    Returns:
        Dict[str, np.ndarray]: Updated numpy storage dictionary with predictions and additional data.
    """
    start_index = 0
    
    for batch, (X, y, pat, age_us, age_mg) in enumerate(dataloader):
        length = len(y)
        np_storage['labels'][start_index:start_index+length] = y.numpy().reshape(-1, 1)
        np_storage['pat_ids'][start_index:start_index+length] = pat.numpy().reshape(-1, 1)

        for name, model in zip(arch_names, archs):
            np_storage[name + '_preds'][start_index:start_index+length] = softmax_obj(model(X).detach()).numpy()

        np_storage['age_us'][start_index:start_index+length] = age_us.numpy().reshape(-1, 1)
        np_storage['age_mg'][start_index:start_index+length] = age_mg.numpy().reshape(-1, 1)

        start_index += length

    return np_storage

def load_checkpoint_and_create_model(file_path: str, arch: str) -> Module:
    """
    Loads a checkpoint from a file and creates a model with the specified architecture.

    Args:
        file_path (str): Path to the checkpoint file.
        arch (str): Architecture of the model ('resnet', 'densenet', 'vit').

    Returns:
        Module: The created and initialized model.
    """
    renamed_checkpoint: Dict[str, Any] = dict()
    checkpoint: Dict[str, Any] = torch.load(file_path)['state_dict']
    renamed_checkpoint['conv1to3.weight'] = checkpoint['model.conv1to3.weight']
    renamed_checkpoint['conv1to3.bias'] = checkpoint['model.conv1to3.bias']

    for key in checkpoint.keys():
        if 'model.model' in key:
            renamed_checkpoint[key[6:]] = checkpoint[key]

    model = CustomNet(lr=0.1, batch=32, arch=arch)
    model.load_state_dict(renamed_checkpoint)
    model.eval()

    return model

if __name__ == '__main__':
    densenet = load_checkpoint_and_create_model(file_path="checkpoints/densenet_checkpoint.ckpt", arch='densenet')
    vit32 = load_checkpoint_and_create_model(file_path="checkpoints/vit_checkpoint.ckpt", arch='vit')
    resnet = load_checkpoint_and_create_model(file_path="checkpoints/resnet_checkpoint.ckpt", arch='resnet')
    
    test_augs = T.Compose([T.ToTensor(), MinMaxNormalization(), CropSquareResize(IM_SIZE=IM_SIZE)])

    val_a_loader, val_b_loader, test_loader, val_a_np, val_b_np, test_np = get_dataloaders(512, test_augs, create_numpy=True, 
                                                                                        archs=['densenet', 'vit', 'resnet'])

    start_index = 0
    m = nn.Softmax(dim=1)

    val_a_np = generate_predictions(val_a_loader, val_a_np, m, [densenet, vit32, resnet], ['densenet', 'vit', 'resnet'])
    with open('predictions_validation_a.pkl', 'wb') as file:
        pickle.dump(val_a_np, file)
    print('completed dirty validation')

    val_b_np = generate_predictions(val_b_loader, val_b_np, m, [densenet, vit32, resnet], ['densenet', 'vit', 'resnet'])
    with open('predictions_validation_b.pkl', 'wb') as file:
        pickle.dump(val_b_np, file)
    print('completed clean validation')

    test_np = generate_predictions(test_loader, test_np, m, [densenet, vit32, resnet], ['densenet', 'vit', 'resnet'])
    with open('predictions_testing.pkl', 'wb') as file:
        pickle.dump(test_np, file)
    print('completed testing')
