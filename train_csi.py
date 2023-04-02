# python ./train_csi.py ./configs/csi.yaml --output reproduce
# python ./eval_csi.py ./configs/csi.yaml ./ckpt/csi_reproduce
# python imports
# 100epochs
# |tIoU = 0.30: mAP = 14.09 (%) Recall@1x = 30.18 (%) Recall@5x = 92.41 (%) 
# |tIoU = 0.40: mAP = 12.86 (%) Recall@1x = 26.98 (%) Recall@5x = 82.90 (%) 
# |tIoU = 0.50: mAP = 8.38 (%) Recall@1x = 16.47 (%) Recall@5x = 67.70 (%) 
# |tIoU = 0.60: mAP = 7.12 (%) Recall@1x = 11.99 (%) Recall@5x = 60.89 (%) 
# |tIoU = 0.70: mAP = 1.66 (%) Recall@1x = 5.18 (%) Recall@5x = 21.98 (%) 

# 35epochs
# |tIoU = 0.30: mAP = 14.16 (%) Recall@1x = 30.18 (%) Recall@5x = 92.05 (%) 
# |tIoU = 0.40: mAP = 12.83 (%) Recall@1x = 26.98 (%) Recall@5x = 84.46 (%) 
# |tIoU = 0.50: mAP = 8.04 (%) Recall@1x = 15.95 (%) Recall@5x = 71.94 (%) 
# |tIoU = 0.60: mAP = 6.96 (%) Recall@1x = 11.99 (%) Recall@5x = 61.09 (%) 
# |tIoU = 0.70: mAP = 1.52 (%) Recall@1x = 5.18 (%) Recall@5x = 20.37 (%) 

# stride16->128 buxing
# |tIoU = 0.30: mAP = 13.17 (%) Recall@1x = 30.02 (%) Recall@5x = 93.16 (%) 
# |tIoU = 0.40: mAP = 12.30 (%) Recall@1x = 26.56 (%) Recall@5x = 86.43 (%) 
# |tIoU = 0.50: mAP = 8.30 (%) Recall@1x = 15.54 (%) Recall@5x = 72.64 (%) 
# |tIoU = 0.60: mAP = 7.39 (%) Recall@1x = 11.99 (%) Recall@5x = 62.64 (%) 
# |tIoU = 0.70: mAP = 2.31 (%) Recall@1x = 5.18 (%) Recall@5x = 27.02 (%) 

# truncate thresh0.9-0.1
# |tIoU = 0.30: mAP = 13.21 (%) Recall@1x = 30.02 (%) Recall@5x = 90.15 (%) 
# |tIoU = 0.40: mAP = 10.85 (%) Recall@1x = 18.59 (%) Recall@5x = 82.11 (%) 
# |tIoU = 0.50: mAP = 7.61 (%) Recall@1x = 15.54 (%) Recall@5x = 66.82 (%) 
# |tIoU = 0.60: mAP = 6.90 (%) Recall@1x = 11.99 (%) Recall@5x = 58.65 (%) 
# |tIoU = 0.70: mAP = 1.48 (%) Recall@1x = 5.18 (%) Recall@5x = 23.38 (%) 

# identity->fpn
# |tIoU = 0.30: mAP = 13.73 (%) Recall@1x = 30.02 (%) Recall@5x = 94.05 (%) 
# |tIoU = 0.40: mAP = 12.74 (%) Recall@1x = 26.56 (%) Recall@5x = 89.13 (%) 
# |tIoU = 0.50: mAP = 8.71 (%) Recall@1x = 15.54 (%) Recall@5x = 76.50 (%) 
# |tIoU = 0.60: mAP = 7.27 (%) Recall@1x = 11.99 (%) Recall@5x = 61.22 (%) 
# |tIoU = 0.70: mAP = 2.73 (%) Recall@1x = 5.18 (%) Recall@5x = 32.72 (%) 

#   one n_mha_win_size to -1
# 13 not good

# next max_seq_len 4608->6912

import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core.config_csi import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

# python ./train_csi.py ./configs/csi.yaml --output reproduce
################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    # pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])
    
    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']
    
    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])
    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now !!!!!!!!!!!!not good
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)
    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    # print("max_epochs", max_epochs)
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        # save ckpt once in a while
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
