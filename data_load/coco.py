import os
import nltk
import torch
import torch.utils.data as data

from PIL import Image
from pycocotools.coco import COCO

class CoCoTrainset(data.Dataset):
    '''load training data of coco caption'''
    '''读取coco caption的训练数据'''
    def __init__(self, img_dir, json, transform, cfg):
        '''

        :param img_dir: image directory (图片目录）
        :param json:    caption json file   （caption的json文件）
        :param vocab:   vocabulary  （词表）
        :param cfg:     configure   （配置表）
        :param transform:   process to image    （对图片的处理）
        '''
        self.img_dir = img_dir
        self.coco = COCO(json)
        self.ann_ids = list(self.coco.anns.keys())
        self.vocab = cfg.vocab
        self.cfg = cfg
        self.transform = transform

    def __getitem__(self, index):
        img_dir = self.img_dir
        coco = self.coco
        vocab = self.vocab
        cfg = self.cfg
        ann_id = self.ann_ids[index]

        cap = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(img_dir, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        cap, cap_len = fix_length(cap, vocab, cfg.max_seq_len)
        return image, cap, cap_len

    def __len__(self):
        return len(self.ann_ids)

def fix_length(cap, vocab, fixed_len):
    '''make caption fixed length'''
    '''把caption 变成固定长度'''

    tokens = nltk.tokenize.word_tokenize(str(cap).lower())
    cap = []
    cap.append(vocab('<start>'))
    cap.extend([vocab(token) for token in tokens])
    cap.append(vocab('<end>'))
    cap_len = len(cap)
    cap_tensor = torch.zeros(fixed_len).long()
    if cap_len <= fixed_len:
        cap_tensor[:cap_len] = torch.Tensor(cap)
    else:
        cap_tensor[:cap_len] = torch.Tensor(cap[:fixed_len])
        cap_len = fixed_len

    return cap_tensor, cap_len


def train_collate_fn(data):
    '''according to length of captions to sort'''
    '''按照caption的长度排序'''

    data.sort(key=lambda x: x[2], reverse=True)
    image, cap, cap_len = zip(*data)
    image = torch.stack(image, 0)
    cap = torch.stack(cap, 0)

    return image, cap, cap_len


def train_load(img_dir, json, transform, cfg, batch_size, shuffle, num_workers):
    '''return a iterable training set'''
    '''返回一个可迭代的coco训练集'''

    coco = CoCoTrainset(img_dir, json, transform, cfg)

    data_loader = data.DataLoader(dataset=coco,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=train_collate_fn,
                                  drop_last=True)

    return data_loader


class Valset(data.Dataset):
    '''Load val set of coco caption'''
    '''读取coco caption的验证数据'''

    def __init__(self, img_dir, json, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(json)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        img_dir = self.img_dir
        coco = self.coco
        img_id = self.img_ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(img_dir, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, img_id

    def __len__(self):
        return len(self.img_ids)

def val_load(img_dir, json, transform, batch_size, shuffle, num_workers):
    '''return a iterable val set'''
    '''返回一个可迭代的coco验证集'''

    coco = Valset(img_dir, json, transform)

    data_loader = data.DataLoader(dataset=coco,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  )

    return data_loader

