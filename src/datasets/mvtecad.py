from torch.utils.data import Subset
from PIL import Image
from torchvision import datasets # from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms
import numpy as np
import torch
from torch.utils.data import Dataset

class MVTecAD_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)
        
        class_table = {0 : 'bottle',
                       1 : 'cable',
                       2 : 'capsule',
                       3 : 'carpet',
                       4 : 'grid',
                       5 : 'hazelnut',
                       6 : 'leather',
                       7 : 'metal_nut',
                       8 : 'pill',
                       9 : 'screw',
                       10 : 'tile',
                       11 : 'toothbrush',
                       12 : 'transistor',
                       13 : 'wood',
                       14 : 'zipper'}
        
        self.root += f'/MVTecAD/{class_table[normal_class]}'
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([0])
        self.outlier_classes = [1]

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(5.8363e-08, 1.0603),
                   (1.3908e-07, 1.2238),
                   (-2.6822e-07, 1.2587),
                   (-8.7932e-08, 1.2332),
                   (-1.2282e-07, 1.1644),
                   (1.5895e-07, 1.1917),
                   (1.9868e-07, 0.6119),
                   (-2.9306e-07, 1.1447),
                   (-9.1890e-08, 1.0325),
                   (5.4836e-07, 1.4141),
                   (2.1607e-07, 1.1487),
                   (8.3121e-08, 1.2111),
                   (1.9868e-08, 1.0787),
                   (-1.5895e-07, 0.5214),
                   (-1.5353e-07, 1.2145)]
        
        def imgs_to_tensor(imgs):
            return torch.stack([transforms.ToTensor()(Image.open(img)) for img in imgs])
        
        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        train_transform = transforms.Lambda(imgs_to_tensor)

        def assign_label(paths):
            labels = []
            for path in paths:
                label = 0 if 'good' in path[0] else 1
                labels.append(label)
            return torch.tensor(labels)

        test_transform = transforms.Lambda(lambda paths: assign_label(paths))            

        img_transform = [
                         transforms.Compose([transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                             transforms.Normalize([min_max[normal_class][0]] * 3, [min_max[normal_class][1] - min_max[normal_class][0]] * 3)]),
                         transforms.Normalize([min_max[normal_class][0]] * 1, [min_max[normal_class][1] - min_max[normal_class][0]] * 1)
                        ]
        
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        
        # Subset train set to normal class
        train_set = MyMVTecAD(root=self.root, train=True, transform=train_transform,
                              label_transform=test_transform, img_transform=img_transform, target_transform=target_transform)
        train_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyMVTecAD(root=self.root, train=False, transform=train_transform,
                                  label_transform=test_transform, img_transform=img_transform, target_transform=target_transform)

        
        
        
class MyMVTecAD():
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root: str, train=True, transform=None, label_transform=None, img_transform=None, target_transform=None):
        self.root = root
        self.data = None
        self.targets = None
        self.transform = transform
        self.label_transform = label_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        
        if train:
            self.root += '/train'
            self.data = np.array(datasets.ImageFolder(root=self.root).imgs, dtype=object)
            self.targets = torch.tensor(self.data[:,1].astype(np.int))
            self.data = self.transform(self.data[:,0])
        else:
            self.root += '/test'
            self.data = np.array(datasets.ImageFolder(root=self.root).imgs, dtype=object)
            self.targets = self.label_transform(self.data)
            self.data = self.transform(self.data[:,0])
            
        if self.data.dim() == 4:
            self.data = self.data.numpy()
        elif self.data.dim() == 3:
            self.data = self.data.unsqueeze(1).numpy()
        else:
            raise ValueError("올바르지 않은 차원입니다.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)
        img = torch.from_numpy(img)

        if self.img_transform is not None:
            try:
                img = self.img_transform[0](img)
            except RuntimeError as e:
                img = self.img_transform[1](img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
