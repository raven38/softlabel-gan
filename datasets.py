''' Datasets
    This file contains definitions for our CIFAR, ImageFolder, and HDF5 datasets
'''
import pickle
import torch
import h5py as h5
import os
import os.path
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm, trange
from joblib import Parallel, delayed

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from torch.utils.data import DataLoader

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def get_imbalanced_subset_idx(X, y, num_samples=5000, num_classes=10):
    from scipy.stats import lognorm, chi2, pareto
    rs = np.random.RandomState(15)
    if num_classes <= 10:
        cdf = lognorm.cdf(x=np.linspace(0, 3, num_classes+1), s=3)
    elif num_classes <= 100:
        # cdf = chi2.cdf(x=np.linspace(0, 3, num_classes+1), df=2)
        cdf = chi2.cdf(x=np.linspace(0, 3, num_classes+1), df=3)
    else:
        # cdf = chi2.cdf(x=np.linspace(0, 8, num_classes+1), df=3)
        cdf = pareto.cdf(x=np.linspace(1, 4, num_classes+1), b=2)
        # cdf = chi2.cdf(x=np.linspace(0.2, 10, num_classes+1), df=5)
    # normed = 1/np.diff(cdf).sum()*np.diff(cdf)
    # class_samples = (num_samples*normed).astype(int)
    class_samples = (num_samples*np.diff(cdf)).astype(int)
    class_samples = class_samples[rs.choice(np.arange(num_classes), num_classes, replace=False)]
    print(class_samples, flush=True)
    idx = np.concatenate([np.where(np.array(y) == i)[0][:s] for i, s in enumerate(class_samples)], axis=0)
    return idx


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(sorted(os.listdir(dir))):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dogball/xxx.png
        root/dogball/xxy.png
        root/dogball/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, load_in_mem=False,
                 index_filename='imagenet_imgs.npz', **kwargs):
        classes, class_to_idx = find_classes(root)
        # Load pre-computed image directory walk
        if os.path.exists(index_filename):
            print('Loading pre-saved Index file %s...' % index_filename)
            imgs = np.load(index_filename)['imgs']
        # If first time, walk the folder directory and save the
        # results to a pre-computed file.
        else:
            print('Generating  Index file %s...' % index_filename)
            imgs = make_dataset(root, class_to_idx)
            np.savez_compressed(index_filename, **{'imgs': imgs})
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.load_in_mem = load_in_mem

        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data, self.labels = [], []
            for index in tqdm(range(len(self.imgs))):
                path, target = imgs[index][0], imgs[index][1]
                self.data.append(self.transform(self.loader(path)))
                self.labels.append(target)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]
        else:
            path, target = self.imgs[index]
            img = self.loader(str(path))
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(img.size(), target)
        return img, int(target)

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_indices(self, c):
        return [i for i, (path, target) in enumerate(self.imgs) if int(target) == c]


class SoftmaxDataset(ImageFolder):
    def __init__(self, root, ann_file=None,
                 target_type='onehot',
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 load_in_mem=False,
                 index_filename='imagenet_imgs.npz',
                 num_samples=None,
                 train=True,
                 **kwargs):
        super(SoftmaxDataset, self).__init__(root, transform=transform, target_transform=target_transform, loader=loader, load_in_mem=False, index_filename=index_filename, **kwargs)
        if target_type in ['softmax', 'ols'] and train is True:
            anns = np.load(ann_file)['arr_0']
            keys = np.load(ann_file)['arr_1']
            anns = {key: ann for key, ann in zip(keys, anns)}
            instance = []
            for target_softmax, _ in self.imgs:
                instance.append(anns['/'.join(target_softmax.split('/')[-2:])])
            self.softmax = instance
            print('labels accuracy', np.sum(np.argmax(self.softmax, axis=1) == np.array([int(im[1]) for im in self.imgs])) / len(self.imgs))
            print(np.argmax(self.softmax, axis=1)[:10])
            print(np.array([im[1] for im in self.imgs])[:10])
            assert len(self.softmax) == len(self.imgs), \
                f'self.softmax and self.targets must be same size' \
                f'len(self.softmax) is {len(self.softmax)}' \
                f'len(self.imgs) is {len(self.imgs)}'

            assert target_type in ['onehot', 'softmax', 'smooth', 'ols'], \
                f'target_type only accept onehot, softmax,' \
                f' and smooth types. given target_type is {target_type}'

        self.train = train
        if train and num_samples is not None and num_samples > 0:
            imb_idx = sorted(get_imbalanced_subset_idx(None, [int(y) for x, y in self.imgs], num_samples=num_samples, num_classes=len(self.classes)))
            self.imgs = [self.imgs[i] for i in imb_idx]
            if target_type in ['softmax', 'ols']:
                self.softmax = [self.softmax[i] for i in imb_idx]
        
        if self.train and target_type == 'ols':
            accumulate_probability = np.zeros((len(self.classes), len(self.classes)))
            class_freq = np.zeros(len(self.classes))
            for idx in range(len(self.imgs)):
                target_class = int(self.imgs[idx][1])
                target_softmax = self.softmax[idx]
                if target_class == np.argmax(target_softmax):
                    accumulate_probability[target_class] += target_softmax
                    class_freq[target_class] += 1
            self.ols_target = accumulate_probability / class_freq

        self.target_type = target_type
        self.smoothing = 0.1 if 'smooth_alpha' not in kwargs else kwargs['smooth_alpha']
        print(f'smooth alpha is setted: {self.smoothing}')

        self.load_in_mem = load_in_mem
        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data = Parallel(n_jobs=512, backend='threading', verbose=1)([delayed(self.loader)(x) for x, y in self.imgs])
            print('Loaded', flush=True)
            # self.data = []
            # for index in tqdm(range(len(self.imgs))):
            #     path = self.imgs[index][0]
            #     self.data.append(self.loader(path))

    def __getitem__(self, index: int):
        path, target_class = self.imgs[index]
        target_class = int(target_class)
        if self.train is False or self.target_type == 'onehot':
            target = np.zeros(len(self.classes))
            target[target_class] = 1
        if self.target_type == 'smooth':
            target = np.ones(len(self.classes)) * (self.smoothing / len(self.classes))
            target[target_class] = 1.-self.smoothing
        if self.train is True and self.target_type == 'ols':
            target = self.ols_target[target_class]
        if self.train is True and self.target_type == 'softmax':
            target = self.softmax[index]
        sample = self.data[index] if self.load_in_mem else self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target.astype(np.float32)

''' ILSVRC_HDF5: A dataset to support I/O from an HDF5 to avoid
    having to load individual images all the time. '''


class ILSVRC_HDF5(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 load_in_mem=False, train=True, download=False, seed=0,
                 val_split=0, **kwargs):  # last four are dummies

        self.root = root
        self.num_imgs = len(h5.File(root, 'r')['labels'])

        # self.transform = transform
        self.target_transform = target_transform

        # Set the transform here
        self.transform = transform

        # load the entire dataset into memory?
        self.load_in_mem = load_in_mem

        # If loading into memory, do so now
        if self.load_in_mem:
            print('Loading %s into memory...' % root)
            with h5.File(root, 'r') as f:
                self.data = f['imgs'][:]
                self.labels = f['labels'][:]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # If loaded the entire dataset in RAM, get image from memory
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]

        # Else load it from disk
        else:
            with h5.File(self.root, 'r') as f:
                img = f['imgs'][index]
                target = f['labels'][index]

        # if self.transform is not None:
                # img = self.transform(img)
        # Apply my own transform
        img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, int(target)

    def __len__(self):
        return self.num_imgs
        # return len(self.f['imgs'])



class CIFAR10(dset.CIFAR10):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=True, seed=0,
                 val_split=0, load_in_mem=True, num_samples=None, **kwargs):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.val_split = val_split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        self.data = []
        self.labels = []
        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.labels += entry['labels']
            else:
                self.labels += entry['fine_labels']
            fo.close()

        self.data = np.concatenate(self.data)
        # Randomly select indices for validation
        if self.val_split > 0:
            label_indices = [[] for _ in range(max(self.labels)+1)]
            for i, l in enumerate(self.labels):
                label_indices[l] += [i]
            label_indices = np.asarray(label_indices)

            # randomly grab 500 elements of each class
            np.random.seed(seed)
            self.val_indices = []
            for l_i in label_indices:
                self.val_indices += list(l_i[np.random.choice(len(l_i), int(
                    len(self.data) * val_split) // (max(self.labels) + 1), replace=False)])

        if self.train == 'validate':
            self.data = self.data[self.val_indices]
            self.labels = list(np.asarray(self.labels)[self.val_indices])

            self.data = self.data.reshape(
                (int(50e3 * self.val_split), 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        elif self.train:
            if self.val_split > 0:
                self.data = np.delete(self.data, self.val_indices, axis=0)
                self.labels = list(np.delete(np.asarray(
                    self.labels), self.val_indices, axis=0))

            self.data = self.data.reshape(
                (int(50e3 * (1.-self.val_split)), 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

            if num_samples is not None:
                np.random.seed(seed)
                train_idx = np.random.choice(
                    len(self.data), num_samples, replace=False)
                self.data = self.data[train_idx]
                self.labels = list(np.asarray(self.labels)[train_idx])
            print(self.data.shape)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.data = entry['data']
            if 'labels' in entry:
                self.labels = entry['labels']
            else:
                self.labels = entry['fine_labels']
            fo.close()
            self.data = self.data.reshape((10000, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_indices(self, c):
        return [i for i, target in enumerate(self.labels) if int(target) == c]


class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]


class SoftmaxCIFAR10(CIFAR10):
    def __init__(self, root, ann_file=None,
                 train=True,
                 target_type='onehot',
                 transform=None,
                 target_transform=None,
                 download=True,
                 seed=0,
                 val_split=0,
                 load_in_mem=False,
                 num_samples=None,
                 **kwargs):
        super(SoftmaxCIFAR10, self).__init__(root, train=train,
                                             transform=transform,
                                             target_transform=target_transform,
                                             download=download,
                                             seed=seed, val_split=val_split,
                                             load_in_mem=load_in_mem,
                                             num_samples=None, **kwargs)

        if target_type in ['softmax', 'ols']:
            if not train:
                ann_root, ann_ext = os.path.splitext(ann_file)
                ann_file = ann_root + '_test' + ann_ext
            anns = np.load(ann_file)['arr_0']
            self.softmax = anns
            assert len(self.softmax) == len(self.data), \
                f'self.softmax and self.targets must be same size' \
                f'len(self.softmax) is {len(self.softmax)}' \
                f'len(self.data) is {len(self.data)}'

        assert target_type in ['onehot', 'softmax', 'smooth', 'ols'], \
            f'target_type only accept onehot, softmax,' \
            f' and smooth types. given target_type is {target_type}'
        self.target_type = target_type
        self.smoothing = 0.1
        self.train = train
        self.num_classes = max(self.labels) + 1
        if train and num_samples is not None and num_samples > 0:
            imb_idx = get_imbalanced_subset_idx(self.data, self.labels, num_samples=num_samples, num_classes=self.num_classes)
            self.data = self.data[imb_idx]
            self.labels = list(np.asarray(self.labels)[imb_idx])
            if target_type in ['softmax', 'ols']:
                self.softmax = self.softmax[imb_idx]

        if target_type == 'ols':
            accumulate_probability = np.zeros((self.num_classes, self.num_classes))
            class_freq = np.zeros(self.num_classes)
            for idx in range(len(self.labels)):
                target_class = int(self.labels[idx])
                target_softmax = self.softmax[idx]
                if target_class == np.argmax(target_softmax):
                    accumulate_probability[target_class] += target_softmax
                    class_freq[target_class] += 1
            self.ols_target = accumulate_probability / class_freq
            # print(self.ols_target)
            # print(self.ols_target.sum(axis=1))

    def __getitem__(self, index: int):
        img, target_class = self.data[index], self.labels[index]
        target_class = int(target_class)
        if self.target_type == 'onehot':
            target = np.zeros(self.num_classes)
            target[target_class] = 1
        if self.target_type == 'smooth':
            target = np.ones(self.num_classes) * (self.smoothing / self.num_classes)
            target[target_class] = 1.-self.smoothing
        
        if self.target_type == 'ols':
            target = self.ols_target[target_class]
        if self.target_type == 'softmax':
            target = self.softmax[index]

        sample = Image.fromarray(img)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target.astype(np.float32)


class SoftmaxCIFAR100(CIFAR100):
    def __init__(self, root, ann_file=None,
                 train=True,
                 target_type='onehot',
                 transform=None,
                 target_transform=None,
                 download=True,
                 seed=0,
                 val_split=0,
                 load_in_mem=False,
                 num_samples=None,
                 **kwargs):
        super(SoftmaxCIFAR100, self).__init__(root, train=train,
                                              transform=transform,
                                              target_transform=target_transform,
                                              download=download,
                                              seed=seed, val_split=val_split,
                                              load_in_mem=load_in_mem,
                                              num_samples=None, **kwargs)

        if target_type in ['softmax', 'ols']:
            if not train:
                ann_root, ann_ext = os.path.splitext(ann_file)
                ann_file = ann_root + '_test' + ann_ext
            anns = np.load(ann_file)['arr_0']
            self.softmax = anns
            assert len(self.softmax) == len(self.data), \
                f'self.softmax and self.targets must be same size' \
                f'len(self.softmax) is {len(self.softmax)}' \
                f'len(self.data) is {len(self.data)}'

        assert target_type in ['onehot', 'softmax', 'smooth', 'ols'], \
            f'target_type only accept onehot, softmax,' \
            f' and smooth types. given target_type is {target_type}'
        self.target_type = target_type
        self.smoothing = 0.1
        self.num_classes = max(self.labels) + 1
        self.train = train
        if train and num_samples is not None and num_samples > 0:
            imb_idx = get_imbalanced_subset_idx(self.data, self.labels, num_samples=num_samples, num_classes=self.num_classes)
            self.data = self.data[imb_idx]
            self.labels = list(np.asarray(self.labels)[imb_idx])
            if target_type in ['softmax', 'ols']:
                self.softmax = self.softmax[imb_idx]

        if target_type == 'ols':
            accumulate_probability = np.zeros((self.num_classes, self.num_classes))
            class_freq = np.zeros(self.num_classes)
            for idx in range(len(self.labels)):
                target_class = int(self.labels[idx])
                target_softmax = self.softmax[idx]
                if target_class == np.argmax(target_softmax):
                    accumulate_probability[target_class] += target_softmax
                    class_freq[target_class] += 1
            self.ols_target = accumulate_probability / class_freq


    def __getitem__(self, index: int):
        img, target_class = self.data[index], self.labels[index]
        target_class = int(target_class)
        if self.target_type == 'onehot':
            target = np.zeros(self.num_classes)
            target[target_class] = 1
        if self.target_type == 'smooth':
            target = np.ones(self.num_classes) * (self.smoothing / self.num_classes)
            target[target_class] = 1.-self.smoothing
        if self.target_type == 'ols':
            target = self.ols_target[target_class]
        if self.target_type == 'softmax':
            target = self.softmax[index]
            # if target.argmax() != target_class:
            #     print(target.argmax(), target_class)
        sample = Image.fromarray(img)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target.astype(np.float32)
