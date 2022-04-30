import os
import os.path as osp
import time
import math
import random
from datetime import timedelta
from argparse import ArgumentParser

import numpy as np
import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import wandb

default_dir = os.environ.get('SM_MODEL_DIR', 'trained_models')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"seed : {seed}")


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default="")

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume-from', type=str, default=None)
    
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--no-val')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--seed', type=str, default=2022)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, log_interval, name, seed, no_val, resume_from):
    # Seed 설정
    set_seed(seed)

    # Dataset Load
    dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model_dir = os.path.join(default_dir, model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model define
    model = EAST()
    model.to(device)
    if resume_from:
        ckpt_path = osp.join(resume_from, 'latest.pth')
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print("model Loaded Successfully!")
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    best_mean_loss = float('inf')
    mean_loss = None
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    else:
        raise Exception("Folder exist. Please Change model_dir. 폴더 중복입니다. model_dir 새로 지정해주세요")

    #################################################
    # wandb 넣어줘야할 부분
    wandb.init(project="", entity="", name=args.name)
    #################################################
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": max_epoch,
        "batch_size": batch_size,
        "image_size": image_size,
        "input_size": input_size,
        "seed": seed
    }
    wandb.watch(model)
    
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        # train
        with tqdm(train_loader, total=num_batches) as pbar:
            for step, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(pbar):
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                train_dict = {
                    'lr': scheduler.get_last_lr()[0], 'Cls loss': extra_info['cls_loss'],
                    'Angle loss': extra_info['angle_loss'], 'IoU loss': extra_info['iou_loss'],
                }
                pbar.set_postfix(train_dict)
                if (step + 1) % log_interval == 0:
                    log = {
                        "loss": loss.item(),
                        'lr': scheduler.get_last_lr()[0],
                        'Cls loss': extra_info['cls_loss'],
                        'Angle loss': extra_info['angle_loss'],
                        'IoU loss': extra_info['iou_loss'],
                        "epoch": epoch + 1
                    }
                    wandb.log(log, step=epoch * num_batches + step)

        scheduler.step()
        mean_loss = epoch_loss / num_batches

        if (epoch + 1) % save_interval == 0:
            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
        if best_mean_loss > mean_loss:
            best_mean_loss = mean_loss
            ckpt_fpath = osp.join(model_dir, 'best_loss.pth')
            torch.save(model.state_dict(), ckpt_fpath)
    wandb.finish()


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
