'''
We write this code with the help of PyTorch demo:
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    
The dataset is downloaded from https://www.kaggle.com/c/oxford-102-flower-pytorch/data

Effects:
        transforms.Resize((464,464)),
        transforms.RandomRotation(15,),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),

wide_resnet101_2 gives 99.39% validation accuracy
wide_resnet101_2 SpinalNet_VGG FC gives 99.14% validation accuracy
wide_resnet101_2 SpinalNet_ResNet FC gives 99.30% validation accuracy
'''

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

# plt.ion()   # interactive mode

def balanced_subset(dataset, length):
    labels = np.asarray([x[1] for x in dataset.imgs])
    classes = np.unique(labels)
    class_data = []
    class_size = []
    for class_index in classes:
        class_data.append(np.where(labels == class_index)[0])
        class_size.append((sum(labels == class_index), class_index))
    class_size = sorted(class_size)
    indices = []
    indices_size = 0
    for class_index in range(len(classes)):
        class_length = min((length - indices_size) // (len(classes) - class_index), class_size[class_index][0])
        # print(class_data[class_size[class_index][1]])
        class_subset = np.random.choice(class_data[class_size[class_index][1]], class_length, replace=False)
        indices.append(class_subset)
        # print(length, indices_size, class_size[class_index][0], class_length)
        indices_size += class_length
        # print(class_length, class_subset)
        assert class_length == len(np.unique(class_subset))
    indices = np.concatenate(indices, axis=0)
    assert len(indices) == length
    assert len(np.unique(indices)) == length, f"{len(np.unique(indices))}, {length}"
    return torch.utils.data.Subset(dataset, indices)

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1).mean()
        return b


class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, tgt = super(ImageFolderWithPath, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, tgt, path

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((464,464)),
        transforms.RandomRotation(15,),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
    'val': transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
}

data_dir = 'data/animeface-character-dataset'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
train_dataset = balanced_subset(train_dataset, len(train_dataset)//3)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
val_dataset, _ = torch.utils.data.random_split(val_dataset, [len(val_dataset) // 2, len(val_dataset) - len(val_dataset) // 2])
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train']}
image_datasets = {'train': train_dataset, 'val': val_dataset}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8 * torch.cuda.device_count(),
                                              shuffle=True, num_workers=1)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes

device = 'cuda'#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
# imshow(out)#, title=[class_names[x] for x in classes])


#%%

# model_ft = models.vgg19_bn(pretrained=True)
# num_ftrs = model_ft.classifier[0].in_features
model_ft = models.wide_resnet101_2(pretrained=True)
num_ftrs = model_ft.fc.in_features

half_in_size = round(num_ftrs/2)
layer_width = 102 #Small for Resnet, large for VGG
Num_class=176

class SpinalNet_ResNet(nn.Module):
    def __init__(self):
        super(SpinalNet_ResNet, self).__init__()
        
        self.fc_spinal_layer1 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(layer_width*4, Num_class),)

    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,half_in_size:2*half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,half_in_size:2*half_in_size], x3], dim=1))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)
        return x

    
class SpinalNet_VGG(nn.Module):
    def __init__(self):
        super(SpinalNet_VGG, self).__init__()
        
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(half_in_size, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            nn.Dropout(p = 0.5), 
            nn.Linear(layer_width*4, Num_class),)        

    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,half_in_size:2*half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,half_in_size:2*half_in_size], x3], dim=1))
        
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        
        x = self.fc_out(x)
        return x



VGG_fc = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, Num_class)
        )


'''
Changing the fully connected layer to SpinalNet or VGG or ResNet
'''

model_ft.fc = nn.Linear(num_ftrs, Num_class)
# model_ft.fc = SpinalNet_ResNet() #SpinalNet_VGG


def evaluate(model, loader):
    model.eval()
    running_corrects = 0
    data_size = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        data_size += inputs.shape[0]

    model.train()
    acc = running_corrects / data_size
    return acc


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    entropy_criterion = HLoss()
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    save_every = 200
    batch_iter = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss_entopy = entropy_criterion(outputs)
                    loss = loss # - 0.7 * loss_entopy

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    batch_iter += 1
                    # if batch_iter % save_every == 0:
                    #     model.eval()
                    #     acc = evaluate(model, dataloaders['val'])
                    #     print(f'iter {batch_iter}: Acc {acc}')
                    #     predict_iter(model, batch_iter)
                    #     torch.save(model.state_dict(), f'model_animeface_iter{batch_iter}.pth')
                    #     model.train()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # if phase == 'val':
            #     predict_iter(model, batch_iter)
            #     torch.save(model.state_dict(), f'model_animeface_iter{batch_iter}.pth')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                time_elapsed = time.time() - since
                print('Time from Start {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

        print('', flush=True)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def predict_iter(model, batch_iter):
    dataset = ImageFolderWithPath(os.path.join(data_dir, 'train'), data_transforms['train'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=16*torch.cuda.device_count(), shuffle=False, num_workers=1)
    print(f"dataset size: {len(dataset)}")
    predicts = []
    paths = []
    ls = []
    for inputs, labels, path in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(inputs)

        predicts.append(F.softmax(logits, dim=1).cpu().numpy())
        paths.extend(path)
        ls.append(labels.cpu().numpy())
    predicts = np.concatenate(predicts, axis=0)
    ls = np.concatenate(ls, axis=0)
    paths = ['/'.join(path.split('/')[-2:]) for path in paths]
    print(predicts.shape)
    np.savez(f'animeface_softmax_iter{batch_iter}.npz', predicts, paths, ls)



def predict(model):
    dataset = ImageFolderWithPath(os.path.join(data_dir, 'train'), data_transforms['train'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=16*torch.cuda.device_count(), shuffle=False, num_workers=1)
    print(f"dataset size: {len(dataset)}")
    predicts = []
    paths = []
    for inputs, labels, path in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(inputs)

        predicts.append(F.softmax(logits).cpu().numpy())
        paths.extend(path)
    predicts = np.concatenate(predicts, axis=0)
    paths = ['/'.join(path.split('/')[-2:]) for path in paths]
    print(predicts.shape)
    np.savez('animeface_softmax_resnet.npz', predicts, paths)



def extract_features(model):
    dataset = ImageFolderWithPath(os.path.join(data_dir, 'train'), data_transforms['train'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"dataset size: {len(dataset)}")
    predicts = []
    paths = []
    for inputs, labels, path in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(inputs)

        predicts.append(logits.cpu().numpy())
        paths.extend(path)
    predicts = np.concatenate(predicts, axis=0)
    paths = ['/'.join(path.split('/')[-2:]) for path in paths]
    print(predicts.shape)
    np.savez('animeface_features3.npz', predicts, paths)
    

model_ft = model_ft.to(device)
# model_ft = torch.nn.DataParallel(model_ft)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)
torch.save(model_ft.state_dict(), 'model_animeface_resnet.pth')
predict(model_ft)

# model_ft.load_state_dict(torch.load('model_animeface.pth'))
# model_ft.fc.fc_out = nn.Identity()
# extract_features(model_ft)
