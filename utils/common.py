''' common function file '''
''' 常用函数文件 '''

import torch
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def set_seed(seed):
    ''' fex the random seed'''
    '''固定随机种子'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_transform():
    '''handle training images'''
    '''对训练图片的处理'''
    train_transform = transforms.Compose([
        transforms.Resize([400, 400], Image.ANTIALIAS),  # 重置图像分辨率，使用ANTTALIAS抗锯齿方法
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    return train_transform


def val_transform():
    '''handle test and val images'''
    '''对验证和测试图片的处理'''
    val_transform = transforms.Compose([
        transforms.Resize([299, 299], Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    return val_transform


def coco_metrics(ref_json, sen_json):
    '''compute coco metrics'''
    '''计算生成的句子的指标'''

    coco = COCO(ref_json)
    cocorefs = coco.loadRes(sen_json)
    cocoEval = COCOEvalCap(coco, cocorefs)
    # cocoEval.params['image_id'] = cocorefs.getImgIds() # using it when you debug (调试代码的时候使用）
    cocoEval.evaluate()
    return cocoEval.eval.items()
