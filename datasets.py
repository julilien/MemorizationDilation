from typing import Tuple, Any, Optional, Callable
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, CIFAR100, SVHN
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import os
from PIL import Image
from pathlib import Path

from dataset_utils import split_train_and_val_data, CIFAR10_STR, CIFAR100_STR, FASHION_MNIST_STR, MNIST_STR, SVHN_STR

TRAIN_TARGETS_FN = "train_targets.pt"
TRAIN_ORIG_TARGETS_FN = "train_original_targets.pt"
TRAIN_DATA_FN = "train_data.pt"
TRAIN_ORIG_DATA_FN = "train_original_data.pt"
SELECTED_LABELS_FN = "selected_labels.pt"

TEST_DATA_FN = "test_data.pt"
TEST_TARGETS_FN = "test_targets.pt"

TRAIN_ACTIVATIONS_NP_FN = "train_activations.npy"
TRAIN_LABELS_NP_FN = "train_labels.npy"
TEST_ACTIVATIONS_NP_FN = "test_activations.npy"
TEST_LABELS_NP_FN = "test_labels.npy"

# Dataset-specific means and stddevs
IMAGENET_MEAN = [0.4914, 0.4822, 0.4465]
IMAGENET_STDDEV = [0.2023, 0.1994, 0.2010]
MNIST_MEAN = (0.1307,)
MNIST_STDDEV = (0.3081,)


def get_dataset(dataset_name, data_dir, custom=False):
    if dataset_name == CIFAR10_STR:
        logging.debug('Dataset: CIFAR10.')
        trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))

        testset = CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))

        trainset.targets = torch.tensor(trainset.targets)
        testset.targets = torch.tensor(testset.targets)

        num_classes = 10
    elif dataset_name == SVHN_STR:
        logging.debug('Dataset: SVHN.')
        trainset = CustomSVHN(root=data_dir, split="train", download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))

        testset = CustomSVHN(root=data_dir, split="test", download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))
        num_classes = 10
    elif dataset_name == CIFAR100_STR:
        logging.debug('Dataset: CIFAR-100.')
        trainset = CIFAR100(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))

        testset = CIFAR100(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))
        num_classes = 100
    elif dataset_name == FASHION_MNIST_STR:
        logging.debug('Dataset: Fashion-MNIST.')
        if not custom:
            trainset = FashionMNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
            ]))
        else:
            trainset = CustomFashionMNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
            ]))

        testset = FashionMNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
        ]))
        num_classes = 10
    elif dataset_name == MNIST_STR:
        logging.debug('Dataset: MNIST.')
        if not custom:
            trainset = MNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
            ]))
        else:
            trainset = CustomMNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
            ]))

        testset = MNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
        ]))
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset {}.".format(dataset_name))

    return trainset, testset, num_classes


def make_selected_dataset(args, dataset_name, data_dir, batch_size=128, sample_size=None, val_split_prop=None,
                          label_noise=0.0, selected_labels=None, shuffle_val_data=False, four_class_problem=False,
                          num_workers=1):
    assert selected_labels is not None, "Selected labels must be provided as a list of two integers."

    trainset, testset, _ = get_dataset(dataset_name, data_dir)

    # Selected classes
    selected_indices = sum(trainset.targets == i for i in selected_labels).bool()
    trainset.targets = trainset.targets[selected_indices]
    trainset.data = trainset.data[selected_indices]

    selected_indices_test = sum(testset.targets == i for i in selected_labels).bool()
    testset.targets = testset.targets[selected_indices_test]
    testset.data = testset.data[selected_indices_test]

    # Flip classes by chance
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    logging.debug("Label noise: {}".format(label_noise))
    logging.debug("Targets before: {}".format(trainset.targets))

    if four_class_problem:
        if isinstance(label_noise, np.ndarray):
            logging.warning("Sampled label noise is not yet supported for the four class problem!")

        torch.manual_seed(args.seed)

        shuffled_idxs = torch.randperm(trainset.data.shape[0])
        trainset.data = trainset.data[shuffled_idxs]
        trainset.targets = trainset.targets[shuffled_idxs]

        torch.save(trainset.targets, os.path.join(args.save_path, TRAIN_ORIG_TARGETS_FN))
        trainset.targets = torch.where(torch.rand(trainset.targets.size()) < label_noise,
                                       torch.randint(0, len(selected_labels), trainset.targets.size()),
                                       trainset.targets)

        # Now, assign proxy classes randomly
        trainset.targets = torch.where(torch.rand(trainset.targets.size()) < args.fc_noise_degree,
                                       trainset.targets + 2, trainset.targets)

        assert len(selected_labels) == 2, "Currently, only two classes are supported for four class problem."

        num_classes = torch.max(trainset.targets).cpu().numpy().item() + 1
    else:
        torch.manual_seed(args.seed)

        if isinstance(label_noise, np.ndarray):
            # Flip labels
            label_noise_t = torch.tensor(label_noise)
            new_targets = torch.where(torch.rand(trainset.targets.size()) < label_noise_t[trainset.targets],
                                      torch.randint(0, len(selected_labels), trainset.targets.size()),
                                      trainset.targets)

            # Subsample instances to balance (aka stratifying)
            min_class_instances = torch.min(torch.bincount(new_targets))
            retained_labels = []
            for i in selected_labels:
                indices = (new_targets == i).nonzero().squeeze().tolist()
                if not type(indices) == list:
                    indices = list(indices)
                indices = indices[:min_class_instances]
                retained_labels += indices

            torch.save(trainset.targets[retained_labels], os.path.join(args.save_path, TRAIN_ORIG_TARGETS_FN))
            trainset.targets = new_targets

            trainset.targets = trainset.targets[retained_labels]
            trainset.data = trainset.data[retained_labels]
        else:
            torch.save(trainset.targets, os.path.join(args.save_path, TRAIN_ORIG_TARGETS_FN))

            trainset.targets = torch.where(torch.rand(trainset.targets.size()) < label_noise,
                                           torch.randint(0, len(selected_labels), trainset.targets.size()),
                                           trainset.targets)

        num_classes = len(selected_labels)

    # torch.abs(trainset.targets - 1), trainset.targets)
    logging.debug("Targets after: {}".format(trainset.targets))

    # Save targets
    torch.save(trainset.targets, os.path.join(args.save_path, TRAIN_TARGETS_FN))
    torch.save(trainset.data, os.path.join(args.save_path, TRAIN_DATA_FN))
    torch.save(selected_labels, os.path.join(args.save_path, SELECTED_LABELS_FN))

    validation_loader = None

    if sample_size is not None:
        trainloader = subsample_data(trainset, val_split_prop, num_classes, sample_size, batch_size,
                                     num_workers=num_workers)
    else:
        if val_split_prop is not None:
            trainloader, validation_loader = split_train_and_val_data(trainset, args, shuffle=shuffle_val_data)
        else:
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validation_loader, testloader, num_classes


def construct_preloaded_dataset(train_data, train_targets, dataset_name, data_dir, batch_size=128, selected_labels=None,
                                args=None, shuffle_val_data=False, four_class_problem=False):
    assert selected_labels is not None, "Selected labels must be provided as a list of two integers."

    trainset, testset, _ = get_dataset(dataset_name, data_dir, custom=True)

    trainset.data = train_data
    trainset.targets = train_targets

    if four_class_problem:
        num_classes = torch.max(trainset.targets).cpu().numpy().item() + 1
    elif selected_labels is not None:
        num_classes = len(selected_labels)
    else:
        logging.warning("As the selected labels are not properly specified, the number of classes can not be "
                        "determined precisely.")
        num_classes = 2

    valloader = None
    if args is not None and args.val_split_prop is not None and args.val_split_prop > 0.0:
        trainloader, valloader = split_train_and_val_data(trainset, args, shuffle=shuffle_val_data)
    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    selected_indices_test = sum(testset.targets == i for i in selected_labels).bool()
    testset.targets = testset.targets[selected_indices_test]
    testset.data = testset.data[selected_indices_test]

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, valloader, testloader, num_classes


class CustomMNIST(MNIST):
    """
    The original MNIST Vision Dataset object only allows for single integer labels.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(CustomMNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CustomFashionMNIST(FashionMNIST):
    """
    The original FashionMNIST Vision Dataset object only allows for single integer labels.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(CustomFashionMNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CustomSVHN(SVHN):
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)
        self.targets = torch.Tensor(self.labels).long()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def make_reproducible_dataset(args, save_path, val_split_prop=None, label_noise=0.0, eval=False, subselect_classes=None,
                              shuffle_val_data=False, num_workers=1):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainset, testset, num_classes = get_dataset(args.dataset, args.data_dir)

    validation_loader = None

    if label_noise > 0.0:
        # Flipping
        if not eval:
            logging.debug("Targets before: {}".format(trainset.targets))
            torch.save(trainset.targets, save_path + "/train_original_targets.pt")
            if torch.is_tensor(trainset.targets):
                trgt_size = trainset.targets.size()
                trgt_size2 = trgt_size
                old_trgts = trainset.targets

                trainset.targets = torch.where(torch.rand(trgt_size) < label_noise,
                                               torch.randint(0, num_classes, trgt_size2),
                                               old_trgts)
            else:
                trgt_size = len(trainset.targets)

                trainset.targets = np.where(np.random.random(trgt_size) < label_noise,
                                            np.random.randint(0, num_classes, trgt_size),
                                            trainset.targets)

            logging.debug("Targets after: {}".format(trainset.targets))

            torch.save(trainset.targets, os.path.join(save_path, TRAIN_TARGETS_FN))
            torch.save(trainset.data, os.path.join(save_path, TRAIN_DATA_FN))
        else:
            trainset.targets = torch.load(os.path.join(save_path, TRAIN_TARGETS_FN))
            logging.debug("Targets: {}".format(trainset.targets))
            trainset.data = torch.load(os.path.join(save_path, TRAIN_DATA_FN))

    if args.sample_size is not None:
        trainloader = subsample_data(trainset, val_split_prop, num_classes, args.sample_size, args.batch_size,
                                     num_workers)
    else:
        if val_split_prop is not None:
            assert label_noise == 0.0, "No noise label in validation data"

            trainloader, validation_loader = split_train_and_val_data(trainset, args, shuffle=shuffle_val_data)
        else:
            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                                     worker_init_fn=lambda id: np.random.seed(id))

    if subselect_classes is not None:
        selected_indices_test = sum(testset.targets == i for i in subselect_classes).bool()
        testset.targets = testset.targets[selected_indices_test]
        testset.data = testset.data[selected_indices_test]

    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validation_loader, testloader, num_classes


def subsample_data(dataset, val_split_prop, num_classes, sample_size, batch_size, num_workers=1):
    if val_split_prop is not None:
        raise NotImplementedError("val_split_prop not yet implemented for subset sample size.")

    total_sample_size = num_classes * sample_size
    cnt_dict = dict()
    total_cnt = 0
    indices = []
    for i in range(len(dataset)):

        if total_cnt == total_sample_size:
            break

        label = dataset[i][1]
        if label not in cnt_dict:
            cnt_dict[label] = 1
            total_cnt += 1
            indices.append(i)
        else:
            if cnt_dict[label] == sample_size:
                continue
            else:
                cnt_dict[label] += 1
                total_cnt += 1
                indices.append(i)

    indices = torch.tensor(indices)
    return DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(indices), num_workers=num_workers)
