import torch
import random
import numpy as np
import sys
sys.path.append('../..')

import argparse
from models.nic import NIC
from data_load.data_load import data_load
from vocab import Vocabulary
from utils.train_cap import eval_cap
from config.nic import NIC_cfg
from utils.common import set_seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    nic_cfg = NIC_cfg()
    # 固定随机种子
    set_seed(21)
    # 读取数据集和训练
    test_data = data_load(args.dataset, nic_cfg, type='test')
    # 搭建模型
    nic = NIC(nic_cfg).to(device)
    # 读取最好模型参数
    checkpoint_path = nic_cfg.best_model.format(args.model, args.version)
    checkpoint = torch.load(checkpoint_path)
    nic.load_state_dict(checkpoint['model'])

    # 预测
    eval_cap(args, nic_cfg, nic, test_data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='nic', help='模型的名字')
    parser.add_argument('--version', type=str, default='baseline', help='模型的版本')
    parser.add_argument('--dataset', type=str, default='karpathy', help='选择数据集')
    parser.add_argument('--beam_num', type=int, default=3, help='beam search的大小')
    args = parser.parse_args()
    main(args)