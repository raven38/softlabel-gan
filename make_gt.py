from datasets import find_classes, make_dataset, pil_loader
import torch 
from torchvision import transforms
from torchvision import utils as vutils
import random
import os

root_dir = '/home/waltz/data/animeface-character-dataset/thumb'
classes, class_to_idx = find_classes(root_dir)
imgs = make_dataset(root_dir, class_to_idx)

trans = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    ])

num_classes = len(classes)
classes_per_sheet = 50
samples_per_class = 10
for i in range(num_classes // classes_per_sheet):
    timgs = []
    for j in range(classes_per_sheet):
        class_imgs = list(filter(lambda x: x[1] == (i*classes_per_sheet) + j, imgs))
        save_imgs = random.sample(class_imgs, samples_per_class)
        print(len(class_imgs), classes[(i*classes_per_sheet)+j], i*classes_per_sheet + j)
        img_path = [x for x, y in save_imgs]

        x = [trans(pil_loader(p)) for p in img_path]

        x = torch.stack(x, 0)
        timgs.append(x)

    out_im = torch.cat(timgs, 0)
    os.makedirs('gt', exist_ok=True)
    out_name = f'gt/{i}.jpg'

    # vutils.save_image(out_im, out_name, nrow=samples_per_class, normalize=True)
