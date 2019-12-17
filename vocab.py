import nltk
import pickle
import argparse

from collections import Counter
from pycocotools.coco import COCO

class Vocabulary(object):
    '''单词表'''
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    def __call__(self, word):

        # use <unk> to replace unknown word (用<unk>代替未登录词)
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(cap_path, threshold):
    '''build a new vocabulary'''
    '''新建一个单词表'''

    coco = COCO(cap_path)
    counter = Counter()
    ids = coco.anns.keys()

    for id in ids:
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # filter out words less than threshold 过滤掉频率小于门限值的单词
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for word in words:
        vocab.add_word(word)
    return vocab


def main(args):

    vocab = build_vocab(args.cap_path, args.threshold)
    vocab_path = args.vocab_path

    # save the vocabulary to pkl format (把单词表保存成pkl格式)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("*** vocabulary size ：{} ***".format(len(vocab)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cap_path', type=str, default='./data/coco/karpathy_split_train.json',
                        help='caption json file (caption的json文件路径) ')
    parser.add_argument('--vocab_path', type=str, default='./data/coco/vocab.pkl',
                        help='path of vocabulary (单词表保存路径)')
    parser.add_argument('--threshold', type=int, default=5,
                        help='threshold of word frequence 单词频率门限值')
    args = parser.parse_args()
    main(args)