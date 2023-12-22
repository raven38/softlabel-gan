import functools
import fnmatch
import importlib
import inspect
import scipy
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import uuid
from typing import Any, List, Tuple, Union
import torch
import lpips

import dnnlib
import dnnlib.tflib
import utils


def prepare_inception_metrics(dataset, parallel, config, device='cpu'):
    dataset = dataset.strip('_hdf5')
    dnnlib.tflib.init_tf()
    pkl_path = 'inception_features/' + dataset + '_inception_moments.pkl'
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    inception_v3_features = dnnlib.util.load_pkl(
        'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_features.pkl')
    inception_v3_softmax = dnnlib.util.load_pkl(
        'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_softmax.pkl')
    try:
        mu_real, sigma_real = dnnlib.util.load_pkl(pkl_path)
    except:
        print('Calculating inception features for the training set...')
        loader = utils.get_data_loaders(
            **{**config, 'train': False, 'mirror_augment': False,
            'use_multiepoch_sampler': False, 'load_in_mem': False, 'pin_memory': False})[0]
        pool = []
        num_gpus = -1 if device == 'cpu' else torch.cuda.device_count()
        for images, _ in loader:
            images = ((images.numpy() * 0.5 + 0.5)
                      * 255 + 0.5).astype(np.uint8)
            pool.append(inception_v3_features.run(images, # minibatch_size=8*num_gpus,
                                                  num_gpus=num_gpus, assume_frozen=True))
        pool = np.concatenate(pool)
        # np.savez(f'{dataset}_inception_features.npz', pool)
        mu_real, sigma_real = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
        dnnlib.util.save_pkl((mu_real, sigma_real), pkl_path)
        mu_real, sigma_real = dnnlib.util.load_pkl(pkl_path)

    def get_inception_metrics(sample, num_inception_images, num_splits=10, prints=True, use_torch=True):
        pool, logits = accumulate_inception_activations(
            sample, inception_v3_features, inception_v3_softmax, num_inception_images, device)
        IS_mean, IS_std = calculate_inception_score(logits, num_splits)
        mu_fake, sigma_fake = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(
            np.dot(sigma_fake, sigma_real), disp=False)  # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        FID = np.real(dist)
        return IS_mean, IS_std, FID
    return get_inception_metrics


def prepare_class_inception_metrics(dataset, num_claases, parallel, config, device='cpu'):
    dataset = dataset.strip('_hdf5')
    dnnlib.tflib.init_tf()
    inception_v3_features = dnnlib.util.load_pkl(
        'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_features.pkl')
    inception_v3_softmax = dnnlib.util.load_pkl(
        'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_softmax.pkl')

    mu_reals, sigma_reals = [], []
    for i in range(num_claases):
        pkl_path = './inception_features/' + dataset + f'_inception_moments_{i}.pkl'
        os.makedirs(pkl_path, exist_ok=True)
        try:
            mu_real, sigma_real = dnnlib.util.load_pkl(pkl_path)
        except:
            print('Calculating inception features for the training set...')
            loader = utils.get_data_loaders(
                **{**config, 'train': False, 'mirror_augment': False,
                   'use_multiepoch_sampler': False, 'load_in_mem': False, 'pin_memory': False})[0]
            pool = []
            num_gpus = -1 if device == 'cpu' else torch.cuda.device_count()
            for images, _ in loader:
                images = ((images.numpy() * 0.5 + 0.5)
                          * 255 + 0.5).astype(np.uint8)
                pool.append(inception_v3_features.run(images, # minibatch_size=8*num_gpus,
                                                      num_gpus=num_gpus, assume_frozen=True))
            pool = np.concatenate(pool)
            mu_real, sigma_real = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
            dnnlib.util.save_pkl((mu_real, sigma_real), pkl_path)
            mu_real, sigma_real = dnnlib.util.load_pkl(pkl_path)
        mu_reals.append(mu_real)
        sigma_reals.append(sigma_real)

    def get_inception_metrics(sample, num_classes, num_inception_images, num_splits=10, prints=True, use_torch=True):
        IS_means, IS_stds, FIDs = [], [], []
        for i in range(num_claases):
            sample_class = functools.partial(sample, y_=i)
            pool, logits = accumulate_inception_activations(
                sample_class, inception_v3_features, inception_v3_softmax, num_inception_images, device)
            IS_mean, IS_std = calculate_inception_score(logits, num_splits)
            mu_fake, sigma_fake = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
            m = np.square(mu_fake - mu_reals[i]).sum()
            s, _ = scipy.linalg.sqrtm(
                np.dot(sigma_fake, sigma_reals[i]), disp=False)  # pylint: disable=no-member
            dist = m + np.trace(sigma_fake + sigma_reals[i] - 2*s)
            FID = np.real(dist)
            IS_means.append(IS_mean)
            IS_stds.append(IS_stds)
            FIDs.append(FID)
        return IS_means, IS_stds, FIDs
    return get_inception_metrics


def prepare_class_FID(dataset, num_claases, parallel, config, device='cpu'):
    dataset = dataset.strip('_hdf5')
    dnnlib.tflib.init_tf()
    inception_v3_features = dnnlib.util.load_pkl(
        'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_features.pkl')
    # inception_v3_softmax = dnnlib.util.load_pkl(
    #     'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_softmax.pkl')

    mu_reals, sigma_reals = [], []
    for i in range(num_claases):
        pkl_path = './inception_features/' + dataset + f'_inception_moments_{i}.pkl'
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        try:
            mu_real, sigma_real = dnnlib.util.load_pkl(pkl_path)
        except:
            print('Calculating inception features for the training set...')
            loader = utils.get_data_loaders(
                **{**config, 'train': False, 'mirror_augment': False, 'drop_last': False,
                   'use_multiepoch_sampler': False, 'load_in_mem': False, 'pin_memory': False, 'c': i})[0]
            pool = []
            num_gpus = -1 if device == 'cpu' else torch.cuda.device_count()
            for images, _ in loader:
                images = ((images.numpy() * 0.5 + 0.5)
                          * 255 + 0.5).astype(np.uint8)
                pool.append(inception_v3_features.run(images, # minibatch_size=8*num_gpus,
                                                      num_gpus=num_gpus, assume_frozen=True))
            pool = np.concatenate(pool)
            mu_real, sigma_real = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
            dnnlib.util.save_pkl((mu_real, sigma_real), pkl_path)
            mu_real, sigma_real = dnnlib.util.load_pkl(pkl_path)
        mu_reals.append(mu_real)
        sigma_reals.append(sigma_real)

    def get_FID(sample, num_classes, num_inception_images, num_splits=10, prints=True, use_torch=True):
        FIDs = []
        for i in range(num_claases):
            sample_class = functools.partial(sample, y_=i)
            # pool, _ = accumulate_inception_activations(
            #     sample_class, inception_v3_features, inception_v3_softmax, num_inception_images)
            pool = accumulate_inception_features(sample_class, inception_v3_features, num_inception_images, device)
            mu_fake, sigma_fake = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
            m = np.square(mu_fake - mu_reals[i]).sum()
            s, _ = scipy.linalg.sqrtm(
                np.dot(sigma_fake, sigma_reals[i]), disp=False)  # pylint: disable=no-member
            dist = m + np.trace(sigma_fake + sigma_reals[i] - 2*s)
            FID = np.real(dist)
            FIDs.append(FID)
        return FIDs

    return get_FID


def prepare_class_mean_FID(dataset, num_claases, parallel, config, device='cpu'):
    dataset = dataset.strip('_hdf5')
    dnnlib.tflib.init_tf()
    inception_v3_features = dnnlib.util.load_pkl(
        'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_features.pkl')
    # inception_v3_softmax = dnnlib.util.load_pkl(
    #     'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_softmax.pkl')

    mu_reals, sigma_reals = [], []
    for i in range(num_claases):
        pkl_path = './inception_features/' + dataset + f'_inception_moments_{i}.pkl'
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        try:
            mu_real, sigma_real = dnnlib.util.load_pkl(pkl_path)
        except:
            print('Calculating inception features for the training set...')
            loader = utils.get_data_loaders(
                **{**config, 'train': False, 'mirror_augment': False, 'drop_last': False,
                   'use_multiepoch_sampler': False, 'load_in_mem': False, 'pin_memory': False, 'c': i})[0]
            pool = []
            num_gpus = -1 if device == 'cpu' else torch.cuda.device_count()
            for images, _ in loader:
                images = ((images.numpy() * 0.5 + 0.5)
                          * 255 + 0.5).astype(np.uint8)
                pool.append(inception_v3_features.run(images, # minibatch_size=8*num_gpus,
                                                      num_gpus=num_gpus, assume_frozen=True))
            pool = np.concatenate(pool)
            mu_real, sigma_real = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
            dnnlib.util.save_pkl((mu_real, sigma_real), pkl_path)
            mu_real, sigma_real = dnnlib.util.load_pkl(pkl_path)
        mu_reals.append(mu_real)
        sigma_reals.append(sigma_real)

    def get_FID(sample, num_classes, num_inception_images, num_splits=10, prints=True, use_torch=True):
        FIDs = []
        for i in range(num_claases):
            sample_class = functools.partial(sample, y_=i)
            # pool, _ = accumulate_inception_activations(
            #     sample_class, inception_v3_features, inception_v3_softmax, num_inception_images)
            pool = accumulate_inception_features(sample_class, inception_v3_features, num_inception_images, device)
            mu_fake, sigma_fake = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
            m = np.square(mu_fake - mu_reals[i]).sum()
            s, _ = scipy.linalg.sqrtm(
                np.dot(sigma_fake, sigma_reals[i]), disp=False)  # pylint: disable=no-member
            dist = m 
            FID = np.real(dist)
            FIDs.append(FID)
        return FIDs

    return get_FID

def prepare_lpips(parallel, device, config):
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn = loss_fn.to('cuda:0')

    def get_lpips(sample, num_images):
        cnt = 0
        pool = []
        while cnt < num_images:
            images, _ = sample()
            images = images.data.cpu()
            # print(images.min(), images.max(), flush=True)
            # images = (images.data.cpu() - 0.5) / 0.5
            assert -1.0 <= images.min() and images.max() <= 1.0, f'min {images.min()}, max{images.max()}'
            pool.append(images)
            cnt += images.shape[0]
        pool = np.concatenate(pool)
        dists = []
        with torch.no_grad():
            for i in range(cnt):
                for j in range(i+1, cnt):
                    dist = loss_fn.forward(torch.tensor(pool[i:i+1], device='cuda:0'), torch.tensor(pool[j:j+1], device='cuda:0'))
                    dists.append(dist.data.cpu())
        return np.mean(np.array(dists)), np.std(np.array(dists))/np.sqrt(len(dists))

    return get_lpips


def prepare_class_lpips(parallel, device, config):
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn = loss_fn.to('cuda:0')

    def get_lpips(sample, num_classes, num_images):
        metrics = []
        for i in range(num_classes):
            cnt = 0
            pool = []
            while cnt < num_images:
                images, _ = sample(y_=i)
                images = images.data.cpu()
                # images = (images.data.cpu() - 0.5) / 0.5
                assert -1.0 <= images.min() and images.max() <= 1.0, f'min {images.min()}, max{images.max()}'
                pool.append(images)
                cnt += images.shape[0]
            pool = np.concatenate(pool)
            dists = []
            with torch.no_grad(): 
                for i in range(cnt):
                    for j in range(i+1, cnt):
                        dist = loss_fn.forward(torch.tensor(pool[i:i+1], device='cuda:0'), torch.tensor(pool[j:j+1], device='cuda:0'))
                        dists.append(dist.data.cpu())
                metrics.append((np.mean(np.array(dists)), np.std(np.array(dists))/np.sqrt(len(dists))))
        return metrics

    return get_lpips

def accumulate_inception_activations(sample, inception_v3_features, inception_v3_softmax, num_inception_images, device='cpu'):
    pool, logits = [], []
    cnt = 0
    num_gpus = -1 if device == 'cpu' else torch.cuda.device_count()
    while cnt < num_inception_images:
        images, _ = sample()
        images = ((images.cpu().numpy() * 0.5 + 0.5)
                  * 255 + 0.5).astype(np.uint8)
        pool.append(inception_v3_features.run(images, # minibatch_size=8*num_gpus,
                                              num_gpus=num_gpus, assume_frozen=True))
        logits.append(inception_v3_softmax.run(images,# minibatch_size=8*num_gpus,
                                               num_gpus=num_gpus, assume_frozen=True))
        cnt += images.shape[0]
    return np.concatenate(pool), np.concatenate(logits, 0)


def accumulate_inception_features(sample, inception_v3_features, num_inception_images, device='cpu'):
    pool = []
    cnt = 0
    num_gpus = -1 if device == 'cpu' else torch.cuda.device_count()
    while cnt < num_inception_images:
        images, _ = sample()
        images = ((images.cpu().numpy() * 0.5 + 0.5)
                  * 255 + 0.5).astype(np.uint8)
        pool.append(inception_v3_features.run(images, # minibatch_size=8*num_gpus,
                                              num_gpus=num_gpus, assume_frozen=True))
        cnt += images.shape[0]
    return np.concatenate(pool)

def calculate_inception_score(pred, num_splits=10):
    scores = []
    for index in range(num_splits):
        pred_chunk = pred[index * (pred.shape[0] // num_splits):(index + 1) * (pred.shape[0] // num_splits), :]
        kl_inception = pred_chunk * \
            (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
        kl_inception = np.mean(np.sum(kl_inception, 1))
        scores.append(np.exp(kl_inception))
    return np.mean(scores), np.std(scores)


def get_inception_features(dataset, parallel, config, device='cpu'):
    dataset = dataset.strip('_hdf5')
    dnnlib.tflib.init_tf()
    inception_v3_features = dnnlib.util.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_features.pkl')

    print('Calculating inception features for the training set...')
    loader = utils.get_data_loaders(
        **{**config, 'train': False, 'mirror_augment': False,
        'use_multiepoch_sampler': False, 'load_in_mem': False, 'pin_memory': False})[0]
    pool = []
    labels = []
    num_gpus = -1 if device == 'cpu' else torch.cuda.device_count()
    for images, label in loader:
        images = ((images.numpy() * 0.5 + 0.5)
                  * 255 + 0.5).astype(np.uint8)
        pool.append(inception_v3_features.run(images, # minibatch_size=8*num_gpus,
                                              num_gpus=num_gpus, assume_frozen=True))
        labels.append(label.cpu().numpy())
    pool = np.concatenate(pool)
    labels = np.concatenate(labels)

    def get_inception_features_gen(sample, num_inception_images):
        gen_pool = accumulate_inception_features(sample, inception_v3_features, num_inception_images, device)
        return gen_pool

    return pool, labels, get_inception_features_gen





