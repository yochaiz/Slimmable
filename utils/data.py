from os.path import join
from numpy import floor

import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils.preprocess import get_transform

__DATASETS_DEFAULT_PATH = '/media/ssd/Datasets/'


def get_dataset(name, train, transform, target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH):
    root = datasets_path  # '/mnt/ssd/ImageNet/ILSVRC/Data/CLS-LOC' #os.path.join(datasets_path, name)

    if name == 'cifar10':
        cifar_ = datasets.CIFAR10(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        return cifar_

    elif name == 'cifar100':
        cifar_ = datasets.CIFAR100(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        return cifar_

    elif name == 'imagenet':
        if train:
            root = join(root, 'train')
        else:
            root = join(root, 'val')

        return datasets.ImageFolder(root=root, transform=transform, target_transform=target_transform)


def load_data(args):
    # init transforms
    transform = {
        'train': get_transform(args.dataset, augment=True),
        'eval': get_transform(args.dataset, augment=False)
    }

    train_data = get_dataset(args.dataset, train=True, transform=transform['train'], datasets_path=args.data)
    valid_data = get_dataset(args.dataset, train=False, transform=transform['eval'], datasets_path=args.data)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(floor(args.train_portion * num_train))

    train_queue = DataLoader(train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[:split]),
                             pin_memory=True, num_workers=args.workers)

    # search_queue = DataLoader(train_data, batch_size=args.batch_size,
    #                           sampler=SubsetRandomSampler(indices[split:num_train]),
    #                           pin_memory=True, num_workers=args.workers)

    valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    # split search_queue to parts
    nParts = args.alphas_data_parts
    nSamples = num_train - split
    nSamplesPerPart = int(nSamples / nParts)
    startIdx = split
    endIdx = startIdx + nSamplesPerPart
    search_queue = []
    for _ in range(nParts - 1):
        dl = DataLoader(train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[startIdx:endIdx]),
                        pin_memory=True, num_workers=args.workers)
        search_queue.append(dl)
        startIdx = endIdx
        endIdx += nSamplesPerPart
    # last part takes what left
    dl = DataLoader(train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[startIdx:num_train]),
                    pin_memory=True, num_workers=args.workers)
    search_queue.append(dl)

    return train_queue, search_queue, valid_queue
