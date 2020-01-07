import torch
import pickle
import sys
sys.path.append('..')
import copy
import torch.nn as nn
from torchvision.models import Inception3
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_urls = {
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def inception_v3(pretrained=False):

    if pretrained:
        model = Inception()
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        return model

    return Inception3()

class Inception(Inception3):
    '''remove the last layer(fc layer)'''
    ''' 去掉最后一层全链接层'''
    def __init__(self):
        super(Inception, self).__init__()

    def forward(self, x):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        # x = self.fc(x)
        return x


class NICEncoder(nn.Module):
    ''' encoder of cnn to represent images'''
    '''CNN编码器，对图片进行表征'''
    def __init__(self):
        super(NICEncoder, self).__init__()

        self.cnn = inception_v3(pretrained=True)
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.cnn.eval()

    def forward(self, image):
        fea_vec = self.cnn(image)

        return fea_vec

    def fine_tune(self):
        for p in self.cnn.parameters():
            p.requires_grad = True
        self.cnn.train()

class NICDecoder(nn.Module):
    '''decoder of NIC'''
    def __init__(self, cfg):

        super(NICDecoder, self).__init__()

        self.fea_dim = cfg.fea_dim
        self.embed_dim = cfg.embed_dim
        self.hid_dim = cfg.hid_dim
        self.max_seq_len = cfg.max_seq_len
        self.vocab = cfg.vocab
        self.vocab_size = len(self.vocab)
        self.lstmcell = nn.LSTMCell(self.embed_dim, self.hid_dim)
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.fea2hid = nn.Linear(self.fea_dim, self.hid_dim)
        self.fc = nn.Linear(self.hid_dim, self.vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, fea_vec, caption, length):
        '''teach-forcing'''
        batch_size = fea_vec.size(0)
        vocab_size = self.vocab_size
        embeddings = self.embed(caption)
        weights = torch.zeros(batch_size, max(length), vocab_size).to(device)
        h, c = self.fea_init_state(fea_vec)

        for t in range(max(length)):
            h, c = self.lstmcell(embeddings[:, t, :], (h, c))
            pred = self.fc(self.dropout(h))
            weights[:, t, :] = pred

        return weights

    def zero_init_state(self, batch_size):
        '''fill the initial state of LSTM with 0'''
        '''用0初始化LSTM状态'''
        h = torch.zeros(batch_size, self.hid_dim).to(device)
        c = torch.zeros(batch_size, self.hid_dim).to(device)
        return h, c

    def fea_init_state(self, fea_vec):
        '''use image features to initiate state of LSTM'''
        '''用features信息初始化LSTM状态'''
        batch_size = fea_vec.size(0)
        h, c = self.zero_init_state(batch_size)
        fea2hid = self.fea2hid(fea_vec)
        h, c = self.lstmcell(fea2hid, (h, c))
        return h, c

    def beam_search(self, fea_vec, num):
        '''generate sentence by beam search'''
        '''beam search搜索句子'''
        fea_vec = fea_vec.unsqueeze(0)
        # initialize (初始化sentence 和h,c)
        h, c = self.fea_init_state(fea_vec)
        sentences = [[[1], 1]]
        h_cells = [0]
        h_cells[0] = (h, c)

        # generate sentence in maximal length (在最大的长度内生成句子)
        for i in range(self.max_seq_len-1):
            sen_size = len(sentences)
            # candidate sequence and state (候选序列和状态)
            candidate_sen = [copy.deepcopy(sen) for sen in sentences * num]
            candidate_hc = [copy.deepcopy(0) for sen in sentences * num]

            for j in range(sen_size):
                # the input is the output of last step (这一时刻的输入，是上一时刻生成的单词)
                inp = torch.Tensor([sentences[j][0][-1]]).long().to(device)
                inp = self.embed(inp)
                # the state of the squeeze(对应这个序列的状态)
                h_cell = h_cells[j]
                h, c = self.lstmcell(inp, h_cell)
                # compute probability (计算概率)
                preds = F.softmax(self.fc(h), 1)
                # select the words of No.k maximal probability 取前num个概率最大的单词
                preds_id = torch.argsort(preds, 1, descending=True)[0][:num]
                for k in range(num):
                    # save the word id and state (把对应的单词id和状态保存到候选句子和状态中)
                    word_id = int(preds_id[k])
                    candidate_sen[j + sen_size * k][0].append(word_id)
                    candidate_sen[j + sen_size * k][1] *= float(preds[0][word_id])
                    candidate_hc[j + sen_size * k] = (h, c)

            # sort by probability (对概率按照降序进行排序，取索引)
            max_ids = sorted(range(len(candidate_sen)), key=lambda k: candidate_sen[k][1], reverse=True)[:num]
            # select the No.k candidate sentences and states of 取前num个概率的候选句子和状态
            sentences = [candidate_sen[max_id] for max_id in max_ids]
            h_cells = [candidate_hc[max_id] for max_id in max_ids]

            # stop beam search if the maximal probability sentence have the end signal
            # 如果概率最大的句子生成了结束符，就停止beam search

            # todo:  problem: how to handle it if the maximal probability sentence not have the end signal,
            # todo: but another sentence that is not the maximal probability sentence have it.
            # todo: 问题：如果最大概率的句子还没生成完，但是有一个句子已经生成了结束符号，但是排在后面，这种情况应该怎么处理
            if sentences[0][0][-1] == 2:
                break

        # exchange word id to word(把单词id换成单词)
        sentence = []
        ids = []    # including <start>包含<start>
        for id in sentences[0][0]:
            word = self.vocab.idx2word[id]
            ids.append(id)
            if word == '<start>':
                continue
            if word == '<end>':
                break
            sentence.append(word)

        return sentence

class NIC(nn.Module):
    '''NIC model'''

    def __init__(self, cfg):
        super(NIC, self).__init__()

        self.encoder = NICEncoder()
        self.decoder = NICDecoder(cfg)

    def forward(self, image, cap, length):

        fea_vec = self.encoder(image)
        weight = self.decoder(fea_vec, cap, length)

        return weight

    def generate(self, image, beam_num, need_extra=True):
        fea_vec,= self.encoder(image)
        sentence = self.decoder.beam_search(fea_vec, beam_num)
        return sentence


