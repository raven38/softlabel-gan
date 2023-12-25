# softlabel-gan
The repository for "Label Augmentation as Inter-class Data Augmentation for Conditional Image Synthesis with Imbalanced Data," WACV 2024  
[Paper](https://openaccess.thecvf.com/content/WACV2024/html/Katsumata_Label_Augmentation_As_Inter-Class_Data_Augmentation_for_Conditional_Image_Synthesis_WACV_2024_paper.html)

This repo is implemented upon the [BigGAN-PyTorch repo](https://github.com/ajbrock/BigGAN-PyTorch) and [DiffAugment repo](https://github.com/mit-han-lab/data-efficient-gans/). 
The main dependencies are:
- Python 3.7.9 or later
- PyTorch 1.10 or later
- TensorFlow 1.15 with GPU support 


## Setup enviroment

```
conda env create tf_env.yaml
conda activate tf
```

## Training classifier
```bash
python3 SpinalNet/Transfer_Learning_AnimeFace.py
```

## Training

```bash
python train.py --experiment_name animeface --DiffAugment color,translation,cutout --mirror_augment --which_best FID --num_inception_images 5000 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 1000 --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --dataset T128 --G_ch 80 --D_ch 80 --G_depth 1 --D_depth 1 --G_shared --shared_dim 128 --dim_z 120 --hier --ema --use_ema --ema_start 20000 --test_every 4000 --save_every 2000 --target_type softmax --ann_file ~/tinyimagenet128_softmax.npz  --adam_eps 1e-6 --SN_eps 1e-6 --BN_eps 1e-4 --num_worker 32  --load_in_mem --G_eval_mode --num_samples 50000
```

## Evaluation

```bash
python eval.py --experiment_name animeface --network weights/animeface/G_ema_best.pth --num_inception_images 5000 --batch_size 32 --parallel --dataset T128 --G_ch 80 --D_ch 80 --G_depth 1 --D_depth 1 --G_shared --shared_dim 128 --dim_z 120 --hier --ema --use_ema --ema_start 20000 --target_type softmax --ann_file ~/tinyimagenet128_softmax.npz  --adam_eps 1e-6 --SN_eps 1e-6 --BN_eps 1e-4 --num_worker 32  --load_in_mem --G_eval_mode
```

## Acknowledgements

The official TensorFlow implementation of the Inception v3 model for IS and FID calculation is borrowed from the [StyleGAN2 repo](https://github.com/NVlabs/stylegan2).
The SpinalNet is borrowed from the [SpinalNet repo](https://github.com/dipuk0506/SpinalNet).
