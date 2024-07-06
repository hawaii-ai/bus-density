
import torch 
from torch import nn
import torchvision.transforms as T
import torchvision.models as models

class MinMaxNormalization(torch.nn.Module):
    """
        Min-max normalization.

        Args:
            img (Tensor): Input image tensor.

        Returns:
            Tensor: Normalized image tensor with values in the range [0, 1].
    """
    def __init__(self):
        super().__init__()

    def forward(self, img):
        img = img*255.0 
        img_std = (img - torch.min(img)) / (torch.max(img) - torch.min(img)) 

        return img_std

class CropSquareResize(torch.nn.Module):
    """
    A PyTorch module that resizes an image and then crops it to a square of a given size.

    Args:
        random (bool): If True, applies a random crop. If False, applies a center crop.
    """
    def __init__(self, IM_SIZE, random: bool = False) -> None:
        super().__init__()
        self.resize = T.Resize(size=IM_SIZE) 
        self.im_size = IM_SIZE
        if random:
            self.crop = T.RandomCrop(size=(IM_SIZE, IM_SIZE), pad_if_needed=True) 
        else:
            self.crop = T.CenterCrop(size=(IM_SIZE, IM_SIZE)) 
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # If the image dimensions are larger than or equal to the target size, apply resize followed by crop
        if img.size()[1] >= self.im_size or img.size()[2] >= self.im_size:
            return self.crop(self.resize(img))
        else:
            # Otherwise, apply only the crop
            return self.crop(img)

class DropoutFC(nn.Module):
    """
    Args:
        in_features (int): The number of input features.
        p (float, optional): The dropout probability. Default is 0.
    """
    def __init__(self, in_features: int, p: float = 0.0) -> None:
        super(DropoutFC, self).__init__()
        self.dropout = nn.Dropout(p=p)  
        self.linear = nn.Linear(in_features, 4)  # we have four density classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)  
        return self.linear(x) 

class CustomNet(nn.Module):
    """
    A class that integrates the architectures tested in the paper (ResNet, DenseNet, ViT)
    with a dropout and fully connected layer and also adds the 1 -> 3 channel initial convolution

    Args:
        lr (float): Learning rate.
        batch (int): Batch size.
        arch (str): The architecture to use ('resnet', 'densenet', or 'vit').
        freeze (int, optional): Number of layers to freeze. Default is 0.
        p (float, optional): Dropout probability. Default is 0.
    """
    def __init__(self, lr: float, batch: int, arch: str, freeze: int = 0, p: float = 0.0) -> None:
        super(CustomNet, self).__init__()
        self.conv1to3 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        if arch == 'resnet':
            self.model = models.resnet50(weights='IMAGENET1K_V1')
            num_features = self.model.fc.in_features
            self.model.fc = DropoutFC(p=p, in_features=num_features)
        elif arch == 'densenet':
            self.model = models.densenet121(weights='IMAGENET1K_V1')
            num_features = self.model.classifier.in_features
            self.model.classifier = DropoutFC(p=p, in_features=num_features)
        elif arch == 'vit':
            self.model = models.vit_b_32(weights='IMAGENET1K_V1')
            num_features = self.model.heads.head.in_features
            self.model.heads.head = DropoutFC(p=p, in_features=num_features)
        else:
            return None

        self.hparams = {'dropout': p, 'frozen': freeze, 'batch_size': batch, 'learning_rate': lr}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1to3(x)
        return self.model(x)