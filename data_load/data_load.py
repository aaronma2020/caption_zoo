import sys
sys.path.append('..')
import pickle
from data_load.coco import train_load, val_load
from utils.common import train_transform, val_transform
from vocab import Vocabulary

def data_load(dataset, cfg, type):
    '''load data'''
    '''读取数据'''
    if type == 'train':
        if dataset == 'karpathy':
            train_data = train_load(cfg.img_dir, cfg.train_kar_cap, train_transform(), cfg, cfg.batch_size, shuffle=True,
                                    num_workers=cfg.num_workers)
            val_data = val_load(cfg.img_dir, cfg.val_kar_cap, val_transform(), batch_size=1, shuffle=False,
                                num_workers=cfg.num_workers)
            return train_data, val_data, cfg.val_kar_cap

        if dataset == 'coco':
            train_data = train_load(cfg.img_dir, cfg.train_coco_cap, train_transform(), cfg, cfg.batch_size, shuffle=True,
                                    num_workers=cfg.num_workers)
            val_data = val_load(cfg.img_dir, cfg.val_coco_cap, val_transform(), batch_size=1, shuffle=False,
                                num_workers=cfg.num_workers)
            return train_data, val_data

    else:
        if dataset == 'karpathy':
            test_data = val_load(cfg.img_dir, cfg.test_kar_cap, val_transform(), batch_size=1, shuffle=False,
                                 num_workers=cfg.num_workers)
            return test_data

        if dataset == 'coco':
            test_data = val_load(cfg.img_dir, cfg.test_coco_cap, val_transform(), batch_size=1, shuffle=False,
                                 num_workers=cfg.num_workers)
            return test_data