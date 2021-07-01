import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from homura.vision import DATASET_REGISTRY
from homura.vision.data.datasets import VisionSet


class ImageNet(Dataset):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None, download=False):
        super(Dataset, self).__init__()

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'train' if train else 'test'

        self.file_pattern = 'miniImageNet_category_split_train_phase_%s.pickle'                 
        self.data = {}

        with open(os.path.join(self.root, self.file_pattern % self.split), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            self.labels = data['labels']

    def __getitem__(self, item):

        img, target = self.imgs[item], self.labels[item]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        
    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    imagenet = ImageNet('/media/hhd1/yue/byol/miniImageNet', 'test', transform=transform)
    print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)
    
    DATASET_REGISTRY.register_from_dict(
      {'mini_imagenet': 
          VisionSet(ImageNet, "/media/hhd1/yue/byol/miniImageNet", 64,
                    [transforms.ToTensor(),
                     transforms.Normalize((120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0), 
                     (70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0))],
                    [transforms.RandomCrop(84, padding=8),
                     transforms.RandomHorizontalFlip()]),})
    train_loader, test_loader, num_classes = DATASET_REGISTRY('mini_imagenet')(batch_size=64, 
                                                                               train_size=4000,
                                                                               drop_last=True, download=False,
                                                                               return_num_classes=True,
                                                                               num_workers=4)
    print(num_classes)
    print(len(train_loader))
    print(len(test_loader))