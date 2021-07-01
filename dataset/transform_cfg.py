from __future__ import print_function

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from typing import Sequence
import torch
import torch.nn as nn

# mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
# std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
normalize = transforms.Normalize(mean=mean, std=std)

class RotationTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img



# transform_A = [
#     transforms.Compose([
#         lambda x: Image.fromarray(x),
#         transforms.RandomCrop(84, padding=8),
#         transforms.RandomApply(
#             [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#         # transforms.RandomApply([GaussianBlur(9)], p=0.5),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.RandomHorizontalFlip(),
#         lambda x: np.asarray(x),
#         transforms.ToTensor(),
#         normalize,
#     ]),

#     transforms.Compose([
#         lambda x: Image.fromarray(x),
#         transforms.ToTensor(),
#         normalize,
#     ])
# ]


# transform_A = [
#     transforms.Compose([
#         lambda x: Image.fromarray(x),
#         transforms.RandomResizedCrop(84),
#         transforms.RandomHorizontalFlip(),
#         lambda x: np.asarray(x),
#         transforms.ToTensor(),
#         normalize,
#     ]),

#     transforms.Compose([
#         lambda x: Image.fromarray(x),
#         transforms.Resize([92, 92]),
#         transforms.CenterCrop(84),
#         transforms.ToTensor(),
#         normalize,
#     ])
# ]

transform_A = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize,
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize,
    ])
]


transform_B = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomResizedCrop(84, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(92),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    ])
]

transform_C = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        # transforms.Resize(92, interpolation = PIL.Image.BICUBIC),
        transforms.RandomResizedCrop(80),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Lighting(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec']),
        # normalize
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(92),
        transforms.CenterCrop(80),
        transforms.ToTensor(),
        # normalize
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    ])
]

# CIFAR style transformation
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
normalize_cifar100 = transforms.Normalize(mean=mean, std=std)
transform_D = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ])
]


transforms_list = ['A', 'B', 'C', 'D']


transforms_options = {
    'A': transform_A,
    'B': transform_B,
    'C': transform_C,
    'D': transform_D,
}
