import sys
import os
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from transformers import CLIPModel
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from tqdm import tqdm
import config
from clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


def set_val_loader(args, preprocess=None, sample=False):
    root = args.root_dir
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                        std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    kwargs = {'num_workers': 4, 'pin_memory': True}

    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'val')
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'val')
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'val')
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'val')
    
    dataset = datasets.ImageFolder(path, transform=preprocess)
       
    if sample:
        idx_path = os.path.join(args.root_dir, 'sub_dataset_indices', 'id', f'{args.in_dataset}.npy')
        if os.path.exists(idx_path):
            indices = np.load(idx_path)
        else:
            size = int(len(dataset)*0.1) if len(dataset) < 1000 else 100
            # indices = np.random.choice(len(dataset), size=size, replace=False) # 从数据集中随机选择100个样本
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)
            indices = indices[:size]
            np.save(idx_path, indices)
        dataset = Subset(dataset, indices) 
        
    print('length of test_dataset: ', len(dataset))
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return val_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess=None, sample=False):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    root = os.path.join(args.root_dir, 'ImageNet_OOD_dataset')
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    if out_dataset == 'ImageNet-O':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'imagenet-o'), transform=preprocess)
    elif out_dataset == 'openimage_o':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'openimage_o'), transform=preprocess)
    elif out_dataset == 'ImageNetOOD':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'ImageNetOOD'), transform=preprocess)
    elif out_dataset == 'ninco':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'NINCO', 'NINCO_OOD_classes'), transform=preprocess)
    elif out_dataset == 'ImageNet10':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'val'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
    elif out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':  # filtered places
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Places'), transform=preprocess)
    elif out_dataset == 'placesbg':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'placesbg'), transform=preprocess)
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'), transform=preprocess)
    elif out_dataset == 'ImageNet-A':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'ImageNet-A'), transform=preprocess)
    elif out_dataset == 'ImageNet-R':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'ImageNet-R'), transform=preprocess)
    elif out_dataset == 'ImageNet-Sketch':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'ImageNet-Sketch'), transform=preprocess)
    elif out_dataset == 'ssb_hard':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'ssb_hard'), transform=preprocess)
    
    if sample:
        idx_path = os.path.join(args.root_dir, 'sub_dataset_indices', 'ood', f'{out_dataset}.npy')
        if os.path.exists(idx_path):
            indices = np.load(idx_path)
        else:
            size = int(len(testsetout)*0.1) if len(testsetout) < 1000 else 100
            indices = np.random.choice(len(testsetout), size=size, replace=False) # 从数据集中随机选择100个样本
            np.save(idx_path, indices)
        testsetout = Subset(testsetout, indices) 

    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    return testloaderOut


def set_train_loader(args, subset=False, max_count=0):  # transformation for CoOp
    root = args.root_dir
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    shuffle = True
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'train')
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')

    dataset = datasets.ImageFolder(path, transform=preprocess)

    if subset:     # random select max_count samples
        from collections import defaultdict
        indices = []
        print('get dataset index')
        classwise_idx = defaultdict(list)
        for i, target in enumerate(tqdm(dataset.targets)):
            classwise_idx[target].append(i)
        print('sample few shot dataset')
        for i in tqdm(range(args.n_cls)):
            sl = np.random.choice(classwise_idx[i], max_count)
            indices.extend(sl)
        dataset = torch.utils.data.Subset(dataset, indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
    return train_loader


class RandomCrop(object):
    def __init__(self, n_crop=2):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        self.n_crop = n_crop
        self.random_crop = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        views = [self.random_crop(x).unsqueeze(dim=0) for _ in range(self.n_crop)]
        views = torch.cat(views, dim=0)
        return views

def set_few_shot_loader(args):  # transformation for ID-like
    root = args.root_dir
    data_transform = RandomCrop(args.n_crop)
    # data_transform = RandomCropAndMask(args.n_crop, args.n_crop)
    shuffle = True
    kwargs = {'num_workers': 0, 'pin_memory': True}

    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
        dataset = datasets.ImageFolder(path)
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'train')
        dataset = datasets.ImageFolder(path)
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
        dataset = datasets.ImageFolder(path)
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
        dataset = datasets.ImageFolder(path)

    indices = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    print('get dataset index')
    for i, target in enumerate(tqdm(dataset.targets)):
        classwise_idx[target].append(i)
    print('sample few shot dataset')
    from random import sample
    for i in tqdm(range(args.n_cls)):
        sl = sample(classwise_idx[i], args.n_shot)
        indices.extend(sl)

    # add data_transform during getting datasets
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    
    dataset = torch.utils.data.Subset(dataset, indices)
    few_shot_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, **kwargs)

    return few_shot_loader

