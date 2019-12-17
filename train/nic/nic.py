''' the training file of NIC'''
''' NIC的训练文件 '''
import sys
sys.path.append('../..')
import os
import argparse
import torch
import random
import numpy as np
from utils.common import set_seed
from config.nic import NIC_cfg
from data_load.data_load import data_load
from models.nic import NIC
from utils.train_cap import train_cap
from vocab import Vocabulary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    # set random seed (固定随机种子）
    set_seed(21)

    nic_cfg = NIC_cfg()

    log = nic_cfg.log.format(args.model, args.version)

    if not os.path.exists(log):
        os.makedirs(log)
    log_path = os.path.join(log, 'log.txt')
    log_content = '对训练数据进行图片方面的数据增强'

    with open(log_path, 'w') as f:
        f.write(log_content)

    # load data (读取数据）
    train_data, val_data, val_cap = data_load(args.dataset, nic_cfg, 'train')

    # load model (加载模型）
    nic = NIC(nic_cfg).to(device)
    # training (训练)
    if args.training_type == 'MLE':
        train_cap(args, nic_cfg, nic, train_data, val_data, val_cap)

    print(log_content)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='nic', help='model name(模型的名字)')
    parser.add_argument('--version', type=str, default='debug', help='model version(模型的版本)')
    parser.add_argument('--training_type', type=str, default='MLE', help='choose how to train nic (MLE, SCST) (选择如何训练nic,（MLE, SCST))')
    parser.add_argument('--dataset', type=str, default='karpathy', help=' choose dataset (coco, Flicker) (选择训练集 (coco, Flicker)')
    args = parser.parse_args()
    main(args)

