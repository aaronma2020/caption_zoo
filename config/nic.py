''' the configure of NIC'''
''' NIC的配置文件'''
import sys
sys.path.append('..')
import pickle
from vocab import Vocabulary
class NIC_cfg():

    def __init__(self):

        self.model = 'nic'
        # model parameters (模型参数)
        self.fea_dim = 2048  # features dimension (图片的特征维度）
        self.embed_dim = 512  # embedding dimension (词嵌入维度）
        self.hid_dim = 512  # hidden layer dimension (隐藏层维度）
        self.max_seq_len = 20  # maximum length of sentence (最大句子长度）
        self.en_lr = 1e-5
        self.de_lr = 4e-4   # learning rate of decoder (解码器的学习率）
        self.beam_num = 3
        self.ft_epoch = 21  # when fine tune encoder (开始训练cnn的轮次）
        self.grad_clip = 0.1

        # save parameters (保存参数）
        self.log = '../../log/{}/{}'
        self.checkpoint = '../../checkpoint/{}/{}'
        self.loss = '../../log/{}/{}/loss.csv'
        self.sentence = '../../log/{}/{}/sentence/'
        self.metrics = '../../log/{}/{}/metrics.csv'
        self.best_model = '../../checkpoint/{}/{}/best_model.tar'
        self.eval_log = '../../eval_log/{}/{}'
        self.eval_sen = '../../eval_log/{}/{}'
        self.eval_metrics = '../../eval_log/{}/{}/metrics.csv'


        # coco dataset (coco数据集）
        self.img_dir = '../../data/coco/coco_image2014'
        self.train_coco_cap = '../../data/coco/annotations/captions_train2014.json'
        self.val_coco_cap = '../../data/coco/annotations/captions_val2014.json'
        self.test_coco_cap = '../../data/coco/annotations/captions_test2014.json'


        # karpathy split (karpathy 划分）
        self.train_kar_cap = '../../data/coco/karpathy/karpathy_split_train.json'
        self.val_kar_cap = '../../data/coco/karpathy/karpathy_split_val.json'
        self.test_kar_cap = '../../data/coco/karpathy/karpathy_split_test.json'
        with open('../../data/coco/karpathy/vocab.pkl', 'rb') as f:
            self.vocab = pickle.load(f)



        #  data loader parameters (data loader参数)
        self.batch_size = 80
        self.num_workers = 4

