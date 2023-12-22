import torch
from tqdm import tqdm
import numpy as np
import functools
import os
import inception_tf
import utils
import dnnlib

import prdc

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

    print("Experiment name is {}_eval".format(config['experiment_name']))
    model = __import__(config['model'])
    G = model.Generator(**config).cuda()
    # print(G.state_dict().keys(), flush=True)
    G_batch_size = max(config['G_batch_size'], config['batch_size']) 
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                                device=device, fp16=config['G_fp16'],
                                z_var=config['z_var'])
    get_inception_metrics = inception_tf.prepare_inception_metrics(config['dataset'], config['parallel'], config, device='gpu')
    get_class_fid = inception_tf.prepare_class_FID(config['dataset'], config['n_classes'], config['parallel'], config, device='gpu')
    get_lpips = inception_tf.prepare_lpips(config['parallel'], device, config)
    get_class_lpips = inception_tf.prepare_class_lpips(config['parallel'], device, config)
    G.load_state_dict(torch.load(dnnlib.util.open_file_or_url(config['network'])))
    if config['G_eval_mode']:
        G.eval()
    else:
        G.train()
    
    sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
    sample_y = functools.partial(utils.sample_y, G=G, z_=z_, num_classes=config['n_classes'], config=config)
    IS_list = []
    FID_list = []
    for _ in tqdm(range(config['repeat'])):
        IS, _, FID = get_inception_metrics(sample, config['num_inception_images'], num_splits=10, prints=False)
        IS_list.append(IS)
        FID_list.append(FID)

    if config['repeat'] > 1:
        print('IS mean: {}, std: {}'.format(np.mean(IS_list), np.std(IS_list)))
        print('FID mean: {}, std: {}'.format(np.mean(FID_list), np.std(FID_list)))
    else:
        print('IS: {}'.format(np.mean(IS_list)))
        print('FID: {}'.format(np.mean(FID_list)))

    class_FID = get_class_fid(sample_y, config['n_classes'], config['num_inception_images'], num_splits=10, prints=False)
    lpips = get_lpips(sample, 100)
    class_lpips = get_class_lpips(sample_y, config['n_classes'], 100)
    print('LPIPS: {}'.format(lpips))
    for i, fid_score in enumerate(class_FID):
        print('Class FID: {} {}'.format(i, fid_score))

    print('Class FID mean: {}, std: {}'.format(np.mean(class_FID), np.std(class_FID)))

    for i, lpips_score in enumerate(class_lpips):
        print('Class LPIPS: {} {}'.format(i, lpips_score))

    print('Class LPIPS mean: {}, std: {}'.format(*(lambda x: (np.mean(x), np.std(x)))([m for m, v in class_lpips])))

    real_pool, real_labels, get_features_gen = inception_tf.get_inception_features(config['dataset'], config['parallel'], config, device='gpu')
    gen_features = get_features_gen(sample, config['num_inception_images'])
    metrics = prdc.compute_prdc(real_pool, gen_features, nearest_k=5)
    prc, rec, dns, cvg = metrics["precision"], metrics["recall"], metrics["density"], metrics["coverage"]
    cprcs, crecs, cdnss, ccvgs = [], [], [], []
    print(f'Precision: {prc}')
    print(f'Recall: {rec}')
    print(f'Density: {dns}')
    print(f'Coverage: {cvg}')
    for i in range(config['n_classes']):
        sample_c = functools.partial(utils.sample_y, G=G, z_=z_, num_classes=config['n_classes'], y_=i, config=config)
        cgen_features = get_features_gen(sample_c, config['num_inception_images'])
        cmetrics = prdc.compute_prdc(real_pool[real_labels.argmax(axis=1) == i], cgen_features, nearest_k=5)
        cprc, crec, cdns, ccvg = cmetrics["precision"], cmetrics["recall"], cmetrics["density"], cmetrics["coverage"]
        print(f'Class Precision, Recall, Density, and Coverage: {cprc} {crec} {cdns} {ccvg}')
        cprcs.append(cprc)
        crecs.append(crec)
        cdnss.append(cdns)
        ccvgs.append(ccvg)

    print('Class Precision mean: {}, std: {}'.format(*(lambda x: (np.mean(x), np.std(x)))([m for m in cprcs])))
    print('Class Recall mean: {}, std: {}'.format(*(lambda x: (np.mean(x), np.std(x)))([m for m in crecs])))
    print('Class Density mean: {}, std: {}'.format(*(lambda x: (np.mean(x), np.std(x)))([m for m in cdnss])))
    print('Class Coverage mean: {}, std: {}'.format(*(lambda x: (np.mean(x), np.std(x)))([m for m in ccvgs])))


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    run_eval(config)

if __name__ == '__main__':
    main()
