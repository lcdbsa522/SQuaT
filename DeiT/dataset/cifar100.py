from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import datasets, transforms


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


def get_cifar100_dataloaders(data_folder="./data/CIFAR100", is_instance=False, download=True):

    class CIFAR100Instance(datasets.CIFAR100):
        def __getitem__(self, index):
            if self.train:
                img, target = self.data[index], self.targets[index]
            else:
                img, target = self.data[index], self.targets[index]

            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target, index

    if is_instance:
        train_set = CIFAR100Instance(
            root=data_folder,
            download=download,
            train=True,
            transform=train_transform
        )
    else:
        train_set = datasets.CIFAR100(
            root=data_folder,
            download=download,
            train=True,
            transform=train_transform
        )

    test_set = datasets.CIFAR100(
        root=data_folder,
        download=download,
        train=False,
        transform=test_transform
    )
    
    return train_set, test_set


def get_cifar100_dataloaders_sample(data_folder="./data/CIFAR100", k=4096, mode='exact', 
                                     is_sample=True, percent=1.0, download=True):

    class CIFAR100InstanceSample(datasets.CIFAR100):
        def __init__(self, root, train=True, transform=None, target_transform=None,
                     download=False, k=4096, mode='exact', is_sample=True, percent=1.0): 
            super().__init__(root=root, train=train, download=download,
                           transform=transform, target_transform=target_transform)
            
            self.k = k
            self.mode = mode
            self.is_sample = is_sample

            num_classes = 100
            if self.train:
                num_samples = len(self.data)
                label = self.targets
            else:
                num_samples = len(self.data)
                label = self.targets

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])
        
            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            
            if 0 < percent < 1:
                n = int(len(self.cls_negative[0]) * percent)
                self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                     for i in range(num_classes)]

            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)
         
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            if not self.is_sample:
                return img, target, index
            else:
                if self.mode == 'exact':
                    pos_idx = index
                elif self.mode == 'relax':
                    pos_idx = np.random.choice(self.cls_positive[target], 1)
                    pos_idx = pos_idx[0]
                else:
                    raise NotImplementedError(self.mode)
                replace = True if self.k > len(self.cls_negative[target]) else False 
                neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace) 
                sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx)) 
                return img, target, index, sample_idx

    train_set = CIFAR100InstanceSample(
        root=data_folder,
        download=download,
        train=True,
        transform=train_transform,
        k=k,
        mode=mode,
        is_sample=is_sample,
        percent=percent
    )

    test_set = datasets.CIFAR100(
        root=data_folder,
        download=download,
        train=False,
        transform=test_transform
    )
    
    return train_set, test_set


