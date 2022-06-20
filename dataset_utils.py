import numpy as np
import os
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

_TRAIN_INDICES_PATH = 'train_indices.npy'
_VAL_INDICES_PATH = 'val_indices.npy'

MNIST_STR = "mnist"
FASHION_MNIST_STR = "fashionmnist"
CIFAR10_STR = "cifar10"
CIFAR100_STR = "cifar100"
SVHN_STR = "svhn"


def split_train_and_val_data(raw_trainset, args, shuffle, num_workers=1):
    ds_size = len(raw_trainset)
    indices = list(range(ds_size))
    split = int(np.floor(args.val_split_prop * ds_size))

    full_train_indices_path = os.path.join(args.save_path, _TRAIN_INDICES_PATH)
    full_val_indices_path = os.path.join(args.save_path, _VAL_INDICES_PATH)

    if os.path.exists(full_train_indices_path) and os.path.exists(full_val_indices_path):
        train_indices = np.load(full_train_indices_path)
        val_indices = np.load(full_val_indices_path)
    else:
        # Shuffle indices
        if shuffle:
            np.random.seed(args.seed)
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        # Save generated indices
        np.save(full_train_indices_path, train_indices)
        np.save(full_val_indices_path, val_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    trainloader = DataLoader(raw_trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers)
    valloader = DataLoader(raw_trainset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=num_workers)

    return trainloader, valloader
