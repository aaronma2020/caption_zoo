import sys
sys.path.append('..')
import torch
import copy

import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torchvision.utils as u
class AttEncoder(nn.Module):
    ''' encoder of cnn to represent images'''
    '''CNN编码器，对图片进行表征'''
    def __init__(self):
        super(AttEncoder, self).__init__()

        cnn = models.vgg19_bn(pretrained=True)
        modules = list(list(cnn.children())[:-2][0].children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.avgpool = nn.AvgPool2d(14)
        for p in self.cnn.parameters():
            p.requires_grad = False

    def forward(self, image):

        batch_size = image.size(0)
        features = self.cnn(image)  # (b,512,14,14)
        fea_vec = self.avgpool(features).view(batch_size, -1)  # (b,512)
        fea_maps = features.permute(0, 2, 3, 1).view(batch_size, 196, -1)
        return fea_vec, fea_maps

    def fine_tune(self):
        for c in list(self.cnn.children())[14:]:
            for p in c.parameters():
                p.requires_grad = True

class Attention(nn.Module):
    '''Attention module'''
    '''Attention模块'''
    def __init__(self, embed_dim, hid_dim, att_dim):
        super(Attention, self).__init__()

        self.fea_att = nn.Linear(embed_dim, att_dim)
        self.hid_att = nn.Linear(hid_dim, att_dim)
        self.att = nn.Linear(att_dim, 1)

    def forward(self, fea_maps, hidden):
        fea_att = self.fea_att(fea_maps)  # (b,196,512) -> (b,196,100)
        hid_att = self.hid_att(hidden)  # (b,512) -> (b,100)
        fusion = fea_att + hid_att.unsqueeze(1)  # (b,49,100)
        att = self.att(torch.relu(fusion)).squeeze(2)  # eq4: eti = fatt(ai,ht-1) (b,196)
        alpha = torch.softmax(att, 1)  # eq5: softmax
        z = (fea_maps * alpha.unsqueeze(2)).sum(1)  # (b,512)
        return z, alpha

class AttDecoder(nn.Module):
    def __init__(self,cfg):

        super(AttDecoder, self).__init__()

        self.fea_dim = cfg.fea_dim
        self.embed_dim = cfg.embed_dim
        self.hid_dim = cfg.hid_dim
        self.max_seq_len = cfg.max_seq_len
        self.vocab = cfg.vocab
        self.vocab_size = len(self.vocab)
        self.att_dim = cfg.att_dim

        self.lstmcell = nn.LSTMCell(2*self.embed_dim, self.hid_dim)
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)

        self.fc = nn.Linear(self.hid_dim, self.vocab_size)
        self.dropout = nn.Dropout(0.5)

        self.attention = Attention(self.embed_dim, self.hid_dim, self.att_dim)
        self.init_h = nn.Linear(self.fea_dim, self.hid_dim)
        self.init_c = nn.Linear(self.fea_dim, self.hid_dim)

        # generate a scalar beta to control rate of attention (Don't use it)
        # 产生一个标量beta，来控制attention的比重 (不使用）
        self.fb = nn.Linear(self.hid_dim, 1)


    def forward(self, fea_vec, fea_maps, cap, cap_len):

        h, c = self.fea_init_state(fea_vec)

        batch_size = fea_maps.size(0)
        vocab_size = self.vocab_size
        embeddings = self.embed(cap)

        weights = torch.zeros(batch_size, max(cap_len), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(cap_len), 196).to(device)
        betas = torch.zeros(batch_size, max(cap_len), 1).to(device)
        for t in range(max(cap_len)):
            z, alpha = self.attention(fea_maps, h)  # att:(b,512)
            beta = torch.sigmoid(self.fb(h))
            z = beta * z
            h, c = self.lstmcell(torch.cat([embeddings[:, t, :], z], 1),
                                 (h, c))
            weight = self.fc(self.dropout(h))

            weights[:, t, :] = weight
            alphas[:, t, :] = alpha
            betas [:, t ,:] = beta
        return weights, alphas, betas


    def beam_search(self, fea_vec, fea_maps, num):
        '''generate sentence by beam search'''
        '''beam search搜索句子'''
        # initialize (初始化sentence 和h,c)
        h, c = self.fea_init_state(fea_vec)
        sentences = [[[1], 1]]
        h_cells = [0]
        h_cells[0] = (h, c)
        alphas = [[]]
        betas = [[]]
        # generate sentence in maximal length (在最大的长度内生成句子)
        for i in range(self.max_seq_len-1):
            sen_size = len(sentences)
            # candidate sequence and state (候选序列和状态)
            candidate_sen = [copy.deepcopy(sen) for sen in sentences * num]
            candidate_hc = [copy.deepcopy(0) for sen in sentences * num]
            candidate_alpha = [copy.deepcopy(alpha) for alpha in alphas * num]
            candidate_beta = [copy.deepcopy(beta) for beta in betas * num]

            for j in range(sen_size):
                # the input is the output of last step (这一时刻的输入，是上一时刻生成的单词)
                inp = torch.Tensor([sentences[j][0][-1]]).long().to(device)
                inp = self.embed(inp)
                # the state of the squeeze(对应这个序列的状态)
                h_cell = h_cells[j]

                z, alpha = self.attention(fea_maps, h)
                beta = torch.sigmoid(self.fb(h))
                z = beta * z
                h, c = self.lstmcell(torch.cat([inp, z], 1), h_cell)
                # compute probability (计算概率)
                preds = torch.softmax(self.fc(h), 1)
                # select the words of No.k maximal probability 取前num个概率最大的单词
                preds_id = torch.argsort(preds, 1, descending=True)[0][:num]
                for k in range(num):
                    # save the word id and state (把对应的单词id和状态保存到候选句子和状态中)
                    word_id = int(preds_id[k])
                    candidate_sen[j + sen_size * k][0].append(word_id)
                    candidate_sen[j + sen_size * k][1] *= float(preds[0][word_id])
                    candidate_hc[j + sen_size * k] = (h, c)
                    candidate_alpha[j + sen_size * k].append(alpha.detach().cpu().numpy())
                    candidate_beta[j + sen_size * k].append(beta.detach().cpu().numpy())

            # sort by probability (对概率按照降序进行排序，取索引)
            max_ids = sorted(range(len(candidate_sen)), key=lambda k: candidate_sen[k][1], reverse=True)[:num]
            # select the No.k candidate sentences and states of 取前num个概率的候选句子和状态
            sentences = [candidate_sen[max_id] for max_id in max_ids]
            h_cells = [candidate_hc[max_id] for max_id in max_ids]
            alphas = [candidate_alpha[max_id] for max_id in max_ids]
            betas = [candidate_beta[max_id] for max_id in max_ids]

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

        return sentence, alphas[0], betas[0]



    def fea_init_state(self, fea_vec):
        '''use image features to initiate state of LSTM'''
        '''用features信息初始化LSTM状态'''
        h = self.init_h(fea_vec)
        c = self.init_c(fea_vec)
        return h, c

class Att(nn.Module):

    def __init__(self, cfg):
        super(Att, self).__init__()
        self.encoder = AttEncoder()
        self.decoder = AttDecoder(cfg)

    def forward(self, image, cap, cap_len):
        fea_vec, fea_maps = self.encoder(image)
        weight, alpha, beta = self.decoder(fea_vec, fea_maps, cap, cap_len)

        return weight, alpha, beta
    def generate(self, image, beam_num, need_extra=True):
        fea_vec, fea_maps = self.encoder(image)
        sentence, alpha, beta = self.decoder.beam_search(fea_vec, fea_maps, beam_num)

        if need_extra == True:
            return sentence, alpha, beta
        else:
            return sentence



