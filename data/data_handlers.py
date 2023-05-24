# Based On:
# https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/data/get_dataloader.py

import torch
import torchvision
import torchvision.transforms as transforms
from data.image_preprocessing import *

from data.noisy_cifar import Noisy_CIFAR10, Noisy_CIFAR100

import os
import numpy as np
np.random.seed(42)

aug = {
    "stl10": {
        # values for train+unsupervised combined
        "mean": [0.44087532, 0.42790526, 0.3867924],
        "std": [0.26826888, 0.2610458, 0.2686684],
        "bw_mean":  [0.42709708],
        "bw_std": [0.257203],
    },
    "cifar10": {
        "mean": [0.49139968, 0.48215827, 0.44653124],
        "std": [0.24703233, 0.24348505, 0.26158768],
        "bw_mean": [0.4808616],
        "bw_std": [0.23919088],
    },
    "cifar100": {
        "mean": [0.5070746, 0.48654896, 0.44091788],
        "std": [0.26733422, 0.25643846, 0.27615058],
        "bw_mean": [0.48748648],
        "bw_std": [0.25063065],
    }
}

# Get transformations that are applied to "entire" image

class Crop(object):
    def __call__(self, image_PIL):
        self.image_size, _ = image_PIL.size
        left = random.randint(0, 5)
        upper = random.randint(0, 5)
        right = left + self.image_size - 5
        lower = upper + self.image_size - 5
        image_PIL = image_PIL.crop((left, upper, right, lower))
        image_PIL = image_PIL.resize((self.image_size, self.image_size))
        return image_PIL
    
class Rotate(object):
    def __call__(self, image_PIL):
        degree = random.randint(0, 3) * 90
        image_PIL = image_PIL.rotate(degree)
        return image_PIL

class Cutout(object):
    def __call__(self, image_PIL):
        self.image_size, _ = image_PIL.size
        image_PIL = transforms.ToTensor()(image_PIL)
        size = random.randint(1, int(self.image_size/3))
        x_coord = random.randint(0, self.image_size - size)
        y_coord = random.randint(0, self.image_size - size)
        image_PIL[:, x_coord:x_coord+size, y_coord:y_coord+size] = 0.5
        image_PIL = transforms.ToPILImage()(image_PIL)
        return image_PIL

class Color(object):
    def __call__(self, image_PIL):
        strength = 1.0
        image_PIL = transforms.ColorJitter(brightness=(1-0.8*strength, 1+0.8*strength), contrast=(1-0.8*strength, 1+0.8*strength),
                                           saturation=(1-0.8*strength, 1+0.8*strength), hue=(-0.2*strength, 0.2*strength))(image_PIL)
        return image_PIL
   
def get_transforms(args, eval, aug):
    
    transformations = []

    if not eval:
        transformations.append(transforms.RandomCrop(args.crop_size, args.padding))
    else:
        transformations.append(transforms.CenterCrop(args.crop_size))

    # Resize image after cropping
    if args.image_resize:
        transformations.append(transforms.Resize(args.image_resize))
    
    # Flip image
    if not eval:
        transformations.append(transforms.RandomHorizontalFlip())

    if not eval:
        transformations_dict = {
                'rotate' : Rotate(),
                'color'  : Color(),
                'cutout' : Cutout(),
                'crop'   : Crop()
            }
        
        t1 = args.t1
        t2 = args.t2
        
        if t2 == '':
            transformations.append(transformations_dict[t1])
        else:   
            transformations.append(transformations_dict[t1])
            transformations.append(transformations_dict[t2])

    # Grayscale, Convert to Tensor and Normalize
    if args.gray:
        transformations.append(transforms.Grayscale())
        transformations.append(transforms.ToTensor())
        if eval or not args.patch_aug:
            transformations.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    else:
        #trans.append(transforms.RandomGrayscale(p=0.25)) # As in CPCV2
        transformations.append(transforms.ToTensor())
        if eval or not args.patch_aug:
            transformations.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    
        # If training CPC then patchify, if required also augment
    if not args.fully_supervised:
        if not eval and args.patch_aug:
            transformations.append(PatchifyAugment(gray=args.gray, grid_size=args.grid_size, t1=args.t1, t2=args.t2))
        else:
            transformations.append(Patchify(grid_size=args.grid_size))

    # If patch_aug then normalization goes last
    if not eval and args.patch_aug:
        if args.gray:
            transformations.append(PatchAugNormalize(mean=aug["bw_mean"], std=aug["bw_std"]))
        else:
            transformations.append(PatchAugNormalize(mean=aug["mean"], std=aug["std"]))

    trans = transforms.Compose(transformations)

    s = "Testing" if eval else "Training"
    print(s + ": " + str(trans))

    return trans


def get_stl10_dataloader(args, labeled=False, validate=False):
    data_path = os.path.join("data", "stl10")

    # Define Transforms
    transform_train = transforms.Compose(
        [get_transforms(args, eval=False, aug=aug["stl10"])])
    transform_valid = transforms.Compose(
        [get_transforms(args, eval=True, aug=aug["stl10"])])

    # Get Datasets
    unsupervised_dataset = torchvision.datasets.STL10(
        data_path, split="unlabeled", transform=transform_train, download=args.download_dataset
    )
    train_dataset = torchvision.datasets.STL10(
        data_path, split="train", transform=transform_train, download=args.download_dataset
    )
    test_dataset = torchvision.datasets.STL10(
        data_path, split="test", transform=transform_valid, download=args.download_dataset
    )

    # Get DataLoaders
    try:
        unsupervised_size = args.unsupervised_size
        try:
            unsupervised_size = int(unsupervised_size)
            indices = list(range(len(unsupervised_dataset)))
            np.random.shuffle(indices)
            unsupervised_indices = indices[:unsupervised_size]
            unsupervised_sampler = torch.utils.data.sampler.SubsetRandomSampler(unsupervised_indices)
            unsupervised_loader = torch.utils.data.DataLoader(
		        unsupervised_dataset, batch_size=args.batch_size, sampler=unsupervised_sampler, num_workers=args.num_workers,
		    )
        except:
            # if unsupervised size not an integer, then set to default
            unsupervised_loader = torch.utils.data.DataLoader(
				unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
			)
    except AttributeError:  
        # if argument not specified, set to default (in the train classifier script for example)
        unsupervised_loader = torch.utils.data.DataLoader(
		    unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
		)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Take subset of training data for train_classifier
    try:
        train_size = args.train_size
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
        )
    except AttributeError:  
        # args.train_size is not defined during train_CPC
        # train_loader is not needed during train_CPC
        train_loader = None

    return (unsupervised_loader, train_loader, test_loader)


def get_cifar_dataloader(args, cifar_classes):
    if cifar_classes == 10:
        data_path = os.path.join("data", "cifar10")

        # Define Transforms
        transform_train = transforms.Compose(
            [get_transforms(args, eval=False, aug=aug["cifar10"])])
        transform_valid = transforms.Compose(
            [get_transforms(args, eval=True, aug=aug["cifar10"])])

        # Get Datasets
        unsupervised_dataset = torchvision.datasets.CIFAR10(
            data_path, train=True, transform=transform_train, download=args.download_dataset
        )
        test_dataset = torchvision.datasets.CIFAR10(
            data_path, train=False, transform=transform_valid, download=args.download_dataset
        )

    elif cifar_classes == 100:
        data_path = os.path.join("data", "cifar100")

        # Define Transforms
        transform_train = transforms.Compose(
            [get_transforms(args, eval=False, aug=aug["cifar100"])])
        transform_valid = transforms.Compose(
            [get_transforms(args, eval=True, aug=aug["cifar100"])])

        # Get Datasets
        unsupervised_dataset = torchvision.datasets.CIFAR100(
            data_path, train=True, transform=transform_train, download=args.download_dataset
        )
        test_dataset = torchvision.datasets.CIFAR100(
            data_path, train=False, transform=transform_valid, download=args.download_dataset
        )

        # Noisy Data
        # noise_rate = 0.5
        # noise_type = 'pairflip' # pairflip or symmetric
        # unsupervised_dataset = Noisy_CIFAR100(
        #     root=data_path, train=True, transform=transform_train, download=args.download_dataset, 
        #     noise_type=noise_type, noise_rate=noise_rate
        # )
        # test_dataset = Noisy_CIFAR100(
        #     root=data_path, train=False, transform=transform_valid, download=args.download_dataset, 
        #     noise_type=noise_type, noise_rate=noise_rate
        # )

    else:
        raise Exception("Not a valid number of classes for CIFAR")

    # Get DataLoaders
    try:
        unsupervised_size = args.unsupervised_size
        try:
            unsupervised_size = int(unsupervised_size)
            indices = list(range(len(unsupervised_dataset)))
            np.random.shuffle(indices)
            unsupervised_indices = indices[:unsupervised_size]
            unsupervised_sampler = torch.utils.data.sampler.SubsetRandomSampler(unsupervised_indices)
            unsupervised_loader = torch.utils.data.DataLoader(
		        unsupervised_dataset, batch_size=args.batch_size, sampler=unsupervised_sampler, num_workers=args.num_workers,
		    )
        except:
            # if unsupervised size not an integer, then set to default
            unsupervised_loader = torch.utils.data.DataLoader(
				unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
			)
    except AttributeError:  
        # if argument not specified, set to default (in the train classifier script for example)
        unsupervised_loader = torch.utils.data.DataLoader(
		    unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
		)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Take subset of training data for train_classifier
    try:
        train_size = args.train_size
        indices = list(range(len(unsupervised_dataset)))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            unsupervised_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
        )
    except AttributeError:  
        # args.train_size is not defined during train_CPC
        # train_loader is not needed during train_CPC
        train_loader = None

    return (unsupervised_loader, train_loader, test_loader)


def get_cifar10_dataloader(args):
    unsupervised_loader, train_loader, test_loader = get_cifar_dataloader(
        args, 10)
    return (unsupervised_loader, train_loader, test_loader)


def get_cifar100_dataloader(args):
    unsupervised_loader, train_loader, test_loader = get_cifar_dataloader(
        args, 100)
    return (unsupervised_loader, train_loader, test_loader)


def create_validation_sampler(dataset_size):
    # Creating data indices for training and validation splits:
    validation_split = 0.2
    shuffle_dataset = True

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def calculate_normalization(dataset):
    if dataset == "stl10":
        data_path = os.path.join("data", "stl10")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.STL10(root=data_path, split="train+unlabeled", download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])  # concatenate each channel
        c2 = np.concatenate([np.asarray(train_set[i][0][1]) for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2]) for i in range(len(train_set))])

        train_mean = [np.mean(c1, axis=(0, 1,)), np.mean(c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [np.std(c1, axis=(0, 1)), np.std(c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.STL10(root=data_path, split="train+unlabeled", download=True, transform=train_transform)

        c = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])

        gray_train_mean = [np.mean(c, axis=(0, 1,))]
        gray_train_std = [np.std(c, axis=(0, 1))]

        # [0.44087532, 0.42790526, 0.3867924] [0.26826888, 0.2610458, 0.2686684]
        # [0.42709708] [0.257203]

    elif dataset == "cifar10":
        data_path = os.path.join("data", "cifar10")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])
        c2 = np.concatenate([np.asarray(train_set[i][0][1]) for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2]) for i in range(len(train_set))])

        train_mean = [np.mean(c1, axis=(0, 1,)), np.mean(c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [np.std(c1, axis=(0, 1)), np.std(c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)

        c = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])

        gray_train_mean = [np.mean(c, axis=(0, 1,))]
        gray_train_std = [np.std(c, axis=(0, 1))]

        # [0.49139968, 0.48215827, 0.44653124] [0.24703233, 0.24348505, 0.26158768]
        # [0.4808616] [0.23919088]

    elif dataset == "cifar100":
        data_path = os.path.join("data", "cifar100")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])
        c2 = np.concatenate([np.asarray(train_set[i][0][1]) for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2]) for i in range(len(train_set))])

        train_mean = [np.mean(c1, axis=(0, 1,)), np.mean(c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [np.std(c1, axis=(0, 1)), np.std(c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)

        c = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])

        gray_train_mean = [np.mean(c, axis=(0, 1,))]
        gray_train_std = [np.std(c, axis=(0, 1))]

        # [0.5070746, 0.48654896, 0.44091788] [0.26733422, 0.25643846, 0.27615058]
        # [0.48748648] [0.25063065]

    else:
        raise Exception("Not a valid dataset choice")

    print(train_mean, train_std)
    print(gray_train_mean, gray_train_std)


if __name__ == "__main__":
    calculate_normalization("stl10")
    calculate_normalization("cifar10")
    calculate_normalization("cifar100")
