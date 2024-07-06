import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning as pl
from lightning.pytorch.loggers import CSVLogger

import os
import sys 
import pandas as pd
from PIL import Image
from pprint import pprint

from modules import MinMaxNormalization, CropSquareResize, CustomNet

# optimal parameters selected by optuna in  the paper 
RESNET_PARAMS = {'backbone': 0,
 'batch_size': 512,
 'dropout': 0.9,
 'jitter': 0,
 'lr': 1.187634396419659e-07}

VIT_PARAMS = {'backbone': 1,
 'batch_size': 64,
 'dropout': 0.30000000000000004,
 'jitter': 1,
 'lr': 5.463291317134828e-07}

DENSENET_PARAMS = {'backbone': 4,
 'batch_size': 256,
 'dropout': 0.9,
 'jitter': 0,
 'lr': 5.282716594883346e-07}

IM_SIZE = 224
EPOCHS = 1000
NUM_WORKERS = 1 

torch.set_num_threads(NUM_WORKERS*2) 
torch.manual_seed(1120)
torch.set_float32_matmul_precision("medium")

class BUSDensityDataset(Dataset):
    def __init__(self, annotations_file, len_file, img_dir, split_img_dir, img_size=IM_SIZE, transform=None, target_transform=None):
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
        self.dataframe = pd.read_csv(annotations_file, low_memory=False).loc[:, ['Image_file', 'DENSITY_MG']].sample(n_sample, axis=0, ignore_index=True)
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

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_dataloaders(batch_size, train_augs, test_augs):
    '''
    Helper function to create the dataloaders with specified augmentations and batch size. 
    For the purposes of the sample data in the repository, all datasets are constructed from sample_data.csv
    
    Input: batch_size -> batch size, could be tuned (int)
           train_augs -> train augmentations (torchvision.Compose)
           test_augs -> test/validation augmentations (torchvision.Compose)
           
    Returns: 3-tuple containing training, validation A, and validation B data loaders
    
    '''
    training_data = BUSDensityDataset(annotations_file='sample_data/sample_data.csv', 
                                      img_dir='sample_data/images', 
                                      split_img_dir=None,
                                      img_size=IM_SIZE, transform=train_augs, len_file=5)
    
    validation_data_a = BUSDensityDataset(annotations_file='sample_data/sample_data.csv', 
                                          img_dir='sample_data/images', 
                                          split_img_dir=None,
                                          img_size=IM_SIZE, transform=test_augs, len_file=5)
    
    validation_data_b = BUSDensityDataset(annotations_file='sample_data/sample_data.csv', 
                                          img_dir='sample_data/images', 
                                          split_img_dir=None,
                                          img_size=IM_SIZE, transform=test_augs, len_file=5)
    
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_a_loader = DataLoader(validation_data_a, batch_size=batch_size, num_workers=NUM_WORKERS)
    val_b_loader = DataLoader(validation_data_b, batch_size=batch_size, num_workers=NUM_WORKERS)
    
    return train_loader, val_a_loader, val_b_loader

class SingleTrainer(pl.LightningModule):
    def __init__(self, unet, loss_fn, lr):
        super().__init__()
        self.model = unet
        self.loss_fn = loss_fn
        self.lr = lr
    
    def configure_callbacks(self):
        # early stopping based on the uncurated validation loss
        early_stop = pl.pytorch.callbacks.early_stopping.EarlyStopping(monitor="dirty_val_loss/dataloader_idx_0", 
                                                                       mode="min", 
                                                                       patience=25)
        # this is used to monitor mode collapse and bail out of training if needed
        early_stop_flg = pl.pytorch.callbacks.early_stopping.EarlyStopping(monitor="collapse_flg", 
                                                                           patience=EPOCHS,
                                                                           mode="max", 
                                                                          divergence_threshold=2)
        return [early_stop, early_stop_flg]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        X, y = batch

        # Compute prediction error
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        collapse_flg = torch.unique(pred).size(dim=0)
        self.log("collapse_flg", collapse_flg, sync_dist=True)
        self.log("training_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return {'loss' : loss}
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # training_step defines the train loop.
        X, y = batch

        # Compute prediction error
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        if dataloader_idx == 0:
            self.log("dirty_val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            return {"dirty_val_loss", loss}
        elif dataloader_idx == 1:
            self.log("clean_val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            return {"clean_val_loss", loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.99, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        return ({
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "training_loss",
            },},)   

if __name__ == '__main__':
    gpu = int(sys.argv[1]) # which GPU index you'd like the model to be placed on 
    model_name = sys.argv[2] # should be one of vit, resnet, densenet

    # Generate the model.
    torch.cuda.empty_cache() 
    if model_name == 'vit':
        hparams = VIT_PARAMS
    elif model_name == 'resnet':
        hparams = RESNET_PARAMS
    elif model_name == 'densenet':
        hparams = DENSENET_PARAMS
    else:
        raise(ValueError)
    # keys: 'lr', 'batch_size', 'jitter', 'backbone', 'dropout' -> as per Optuna runs 

    loss_fn = torch.nn.CrossEntropyLoss()

    train_augs = [T.ToTensor(),
                    T.RandomApply([T.RandomRotation(degrees=(0, 360))], p=0.5),
                    MinMaxNormalization(),
                    CropSquareResize(IM_SIZE=IM_SIZE, random=True)]

    # if in Optuna we chose to apply brightness/contrast jitter
    if hparams['jitter']:
        train_augs.append(T.RandomApply([T.ColorJitter(brightness=1, 
                                                    contrast=1, 
                                                    saturation=0, hue=0)], p=0.5))


    train_augs = T.Compose(train_augs)

    test_augs = T.Compose([T.ToTensor(),
                           MinMaxNormalization(),
                           CropSquareResize(IM_SIZE=IM_SIZE)])

    train_loader, val_dirty_loader, val_clean_loader = get_dataloaders(hparams['batch_size'], train_augs, test_augs)

    model = CustomNet(arch=model_name, p=hparams['dropout'], lr=hparams['lr'], batch=hparams['batch_size'], freeze=hparams['backbone'])

    # defining where we freeze the backbones for each of the architectures, depending on which 
    # option was chosen by Optuna during our hyperparameter runs - see supplement of paper 
    if model_name == 'resnet':
        to_freeze = {1 : ['layer1'], 
                    2: ['layer2', 'layer1'], 
                    3: ['layer3', 'layer2', 'layer1'], 
                    4: ['layer4', 'layer3', 'layer2', 'layer1']}
    elif model_name == 'densenet':
        to_freeze = {1 : ['denseblock1'], 
                 2: ['denseblock2', 'denseblock1'], 
                 3: ['denseblock3', 'denseblock2', 'denseblock1'], 
                 4: ['denseblock4', 'denseblock3', 'denseblock2', 'denseblock1']}
    elif model_name == 'vit':
        to_freeze = {1 : ['encoder_layer_0', 'encoder_layer_1', 'encoder_layer_2'], 
                 2: ['encoder_layer_0', 'encoder_layer_1', 'encoder_layer_2', 
                     'encoder_layer_3', 'encoder_layer_4', 'encoder_layer_5'], 
                 3: ['encoder_layer_0', 'encoder_layer_1', 'encoder_layer_2', 
                     'encoder_layer_3', 'encoder_layer_4', 'encoder_layer_5', 
                     'encoder_layer_6', 'encoder_layer_7', 'encoder_layer_8'], 
                 4: ['encoder_layer_0', 'encoder_layer_1', 'encoder_layer_2', 
                     'encoder_layer_3', 'encoder_layer_4', 'encoder_layer_5', 
                     'encoder_layer_6', 'encoder_layer_7', 'encoder_layer_8', 
                     'encoder_layer_9', 'encoder_layer_10', 'encoder_layer_11']}
    else: sys.exit(1) # if you provide an incorrect/unhandled model name 

    # if we selected to freeze any layers, freeze them 
    if hparams['backbone'] > 0:
        for l in to_freeze[hparams['backbone']]:
            for n, p in model.named_parameters():
                if l in n:
                    p.requires_grad = False

    pprint(hparams)

    logger = CSVLogger(model_name+"post_optuna/")
    unet = SingleTrainer(unet=model, loss_fn=loss_fn, lr=hparams['lr'])
    trainer = pl.Trainer(accelerator="gpu", 
                         log_every_n_steps=100,
                         logger=logger,
                         devices=[gpu], 
                         max_epochs=EPOCHS, default_root_dir=model_name+"post_optuna/")
    
    # train and monitor performance
    trainer.fit(unet, train_loader, [val_dirty_loader, val_clean_loader])
    
    score = trainer.validate(unet, dataloaders=[val_dirty_loader, val_clean_loader])
    
    print('Dirty loss: ', score[0]['dirty_val_loss/dataloader_idx_0'])
    print('Clean loss: ', score[0]['clean_val_loss/dataloader_idx_1'])