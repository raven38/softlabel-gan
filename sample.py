import torch
from tqdm import tqdm
import numpy as np
import functools
import os
import inception_tf
import utils
import dnnlib


def run_eval(config):
    # update config (see train.py for explanation)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    device = 'cuda'

    model = __import__(config['model'])
    G = model.Generator(**config).cuda()
    print(G.state_dict().keys(), flush=True)
    G_batch_size = max(config['G_batch_size'], config['batch_size']) 
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                                device=device, fp16=config['G_fp16'],
                                z_var=config['z_var'])

    G.load_state_dict(torch.load(dnnlib.util.open_file_or_url(config['network'])))
    if config['G_eval_mode']:
        G.eval()
    else:
        G.train()

    if config['sample_order'] is None:
        utils.sample_sheet(G,
                           classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                           num_classes=config['n_classes'],
                           samples_per_class=50, parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=config['experiment_name'],
                           folder_number=99999999,
                           z_=z_)

    else:
        # order = [132, 129,  82, 162, 120, 174, 157, 152,  77, 161, 148, 173, 101,
        #          30, 138, 143, 156, 170, 151, 149, 142,  97, 133, 139, 136, 175,
        #          150, 121,  79, 144, 122, 141, 135, 165,  22,  32, 164, 119, 167,
        #          171, 163, 159, 100, 126, 115,  81, 131, 169,   0,  41,  40, 125,
        #          103,  33,  48, 105, 146, 128, 168, 110, 153, 172, 155, 104, 137,
        #          134,  27, 147, 113, 166, 117, 140, 111,   5, 123,  75, 154,   6,
        #          65,  35,  24, 108,  18, 118,  26,  19,  90,  25, 107,  11,  88,
        #          116,  47,   3, 158,  84, 130,  85,  13,  43,  21,  28,  68,  76,
        #          98, 160, 109, 145,  20,  31, 124,  55,  45,  96, 112,  16,  61,
        #          57,  62,   8,  36,  86,  99,  49, 114,  54,  56,  91,  12,  93,
        #          127,  89,  74,   9,  15,   2,  78,  94, 102,   7,  87,  95,  92,
        #          4,  44,  23,  14,  38,  83,  10,  53,  64,  60,  34,  46,  29,
        #          39,  80,  73,  59,  17,  69,  51,  66,  50,  70,  71, 106,  67,
        #          63,   1,  72,  58,  42,  52,  37]

        # order = [  0,  26,  33,  44,   6,  20,   2,  78,  15,  25,  24,  38,  66,
        #            23,  34,   9,  31,   5,   8,  32,  92, 101,  13,  18,  48,  14,
        #            12,  99,  60,  63,  30,  68,  67,  62,  61,  37,   3,  19,  85,
        #            100,  21,  41,   1,  53,  65,  69,  86,  98,  84,   4,  96,  27,
        #            91,  58,  39,  56,  46,  54,  47,  35,  90,  70,  28,  89,  17,
        #            97,  51,  16,  29,   7,  83,  10,  11,  95,  22,  49,  43,  52,
        #            71,  64,  79,  75,  36,  55,  59,  81,  57,  74,  40,  94,  42,
        #            82,  77,  87,  93,  80,  73,  88,  72,  45,  76,  50]

        # order = [184, 59, 72, 48, 8, 44, 67, 181, 93, 113, 1, 142, 194, 9,
        #          79, 177, 92, 102, 36, 187, 7, 25, 90, 119, 188, 175, 168, 65, 160,
        #          81, 134, 105, 47, 56, 29, 91, 193, 69, 189, 4, 21, 172, 140, 3,
        #          109, 114, 51, 60, 139, 120, 130, 49, 167, 70, 106, 156, 54, 173,
        #          34, 147, 143, 190, 80, 166, 78, 41, 191, 154, 15, 117, 43, 83,
        #          186, 33, 19, 127, 57, 112, 170, 128, 131, 138, 2, 185, 151, 136,
        #          115, 118, 164, 66, 94, 95, 125, 24, 75, 159, 145, 40, 152, 97, 55,
        #          6, 32, 179, 148, 42, 58, 155, 18, 141, 135, 82, 171, 176, 63, 46,
        #          174, 100, 13, 5, 162, 88, 165, 153, 85, 86, 110, 53, 108, 133,
        #          104, 180, 74, 192, 61, 11, 73, 23, 30, 14, 64, 62, 76, 163, 182,
        #          158, 35, 121, 50, 84, 169, 87, 89, 161, 132, 101, 28, 111, 122,
        #          10, 183, 146, 45, 149, 126, 37, 26, 68, 27, 150, 157, 129, 116,
        #          39, 20, 17, 107, 99, 123, 22, 71, 16, 38, 96, 52, 98, 144, 77,
        #          103, 124, 137, 0, 12, 31, 178, 195]

        assert os.path.exists(config['sample_order']), 'do not found sample_order file. please specify correct path'
        order = list(np.loadtxt(config['sample_order'], dtype='int'))
        utils.ordered_sample_sheet(G,
                                   classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                                   num_classes=config['n_classes'],
                                   samples_per_class=50, parallel=config['parallel'],
                                   samples_root=config['samples_root'],
                                   experiment_name=config['experiment_name'],
                                   folder_number=99999999,
                                   order=order,
                                   z_=z_)


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    run_eval(config)


if __name__ == '__main__':
    main()
