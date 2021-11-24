import numpy as np
from PIL import Image
from skimage.color import rgb2lab, rgb2ycbcr
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

SIZE = 256


class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train', color_space='Lab'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                # transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
        self.color_space = color_space
        self.split = split
        self.size = SIZE
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        return self.__transform_to_color_space(img) # Converting RGB to L*a*b

    def __transform_to_color_space(self, img):
        if self.color_space == 'Lab':
            img_lab = rgb2lab(img).astype("float32")
            img_lab = transforms.ToTensor()(img_lab)
            known_channel = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
            unknown_channels = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        elif self.color_space == 'HSL':
            img_hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype("float32")
            img_hsl = transforms.ToTensor()(img_hsl)
            h =  img_hsl[[0], ...] / 90. - 1.
            l =  (img_hsl[[1], ...] * 2.) / 255. - 1.
            s =  (img_hsl[[2], ...] * 2.) / 255. - 1.
            known_channel = l # Between -1 and 1
            unknown_channels = torch.cat([h, s]) # Between -1 and 1
        elif self.color_space == "YCbCr":
            img_lab = rgb2ycbcr(img).astype("float32")
            img_lab = transforms.ToTensor()(img_lab)
            known_channel = ((img_lab[[0], ...] - 16.)  *2.)/ 219. - 1. # Between -1 and 1
            unknown_channels = (img_lab[[1, 2], ...] -16.) / 112. - 1. # Between -1 and 1

        return {'known_channel': known_channel, 'unknown_channels': unknown_channels}

    
    def __len__(self):
        return len(self.paths)

def make_dataloaders(batch_size=8, n_workers=0, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader