import os
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from data_load.data_load import data_load
from .save import create_file, create_result, save_loss, save_sentence, save_metrics, save_best_model
from .common import coco_metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def optimize(model, loss, optim, grad_clip=None):
    '''back propagate'''
    '''反向传播'''
    model.zero_grad()
    if grad_clip != None:
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_clip)
    loss.backward()
    optim.step()


def train_cap(args, cfg, model, train_data, val_data, val_cap):
    '''main function of training caption'''
    '''训练caption的循环函数'''

    # create save file (创建保存文件夹)
    loss_path, metrics_path, sen_path, best_model_path = create_file(args.model, args.version, cfg)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=cfg.de_lr)

    best_score = 0
    best_epoch = 0

    for epoch in range(1, 1000):

        # fine turn cnn (开始训练cnn)
        if epoch == cfg.ft_epoch:
            print('*** fine tune cnn ***')
            # load best model within 20 epochs (加载20轮中最好的模型）
            model.load_state_dict((torch.load(best_model_path)['model']))
            optimizer = torch.optim.Adam([{'params': model.decoder.parameters(), 'lr': cfg.de_lr},
                                          {'params': model.encoder.parameters(), 'lr': cfg.en_lr}, ],
                                         betas=(0.8, 0.999))

            model.encoder.fine_tune()
        print('*** epoch:{} ***'.format(epoch))
        # ======= training(训练) ======
        model.train()
        epoch_loss = 0
        total_step = len(train_data)


        for i, (image, cap, cap_len) in tqdm(enumerate(train_data)):
            batch_size = image.size(0)
            image = image.to(device)
            cap = cap.to(device)
            cap_len = [len - 1 for len in cap_len]
            target = pack_padded_sequence(cap[:, 1:], cap_len, batch_first=True)[0]


            if args.model == 'nic':
                weight= model(image, cap, cap_len)
                weight = pack_padded_sequence(weight, cap_len, batch_first=True)[0]
                loss = criterion(weight, target)


            if args.model == 'att':
                weight, alpha, beta = model(image, cap, cap_len)
                weight = pack_padded_sequence(weight, cap_len, batch_first=True)[0]
                loss = criterion(weight, target)

                alpha_loss = torch.sum(torch.pow((1-torch.sum(alpha,1)),2)) / batch_size

                loss += cfg.lam * alpha_loss

            epoch_loss += loss.item()
            optimize(model, loss, optimizer)

        save_loss(epoch_loss / total_step, epoch, loss_path)

        print("*** evaluate val set ***")
        model.eval()
        sentence_list = []
        for i, (image, img_id, img_path) in tqdm(enumerate(val_data)):
            image = image.to(device)
            img_id = img_id[0]

            if args.model == 'nic':
                sentence = model.generate(image, beam_num=cfg.beam_num)
            elif args.model == 'att':
                sentence, alpha, beta = model.generate(image, beam_num=cfg.beam_num)

            sentence = ' '.join(sentence)
            item = {'image_id': int(img_id), 'caption': sentence}
            sentence_list.append(item)

        print('*** compute scores ***')
        sen_json = save_sentence(sentence_list, epoch, sen_path)
        results = coco_metrics(val_cap, sen_json)

        score_dict = save_metrics(results, epoch, metrics_path)
        epoch_score = score_dict['CIDEr']

        # save best model (最好的保存模型)
        if best_score < epoch_score:
            best_score_dict = score_dict
            best_score = epoch_score
            best_epoch = epoch
            save_best_model(model, optimizer, epoch, best_score_dict, best_epoch, best_model_path)

        if (epoch - best_epoch) > 10:
            print(f'total epoch:{epoch} ')
            print(f'complete training best epoch:{best_epoch}, best CIDEr:{best_score}')
            break


def eval_cap(args, cfg, model, test_data, test_cap):
    ''' test caption result'''
    '''测试caption模型'''
    # create save file (创建保存文件夹)
    metrics_path, sen_path = create_result(args.model, args.version, cfg)

    model.eval()
    sentence_list = []
    for i, (image, img_id, path) in tqdm(enumerate(test_data)):
        image = image.to(device)
        img_id = img_id[0]

        sentence = model.generate(image, args.beam_num, need_extra=False)
        sentence = ' '.join(sentence)
        item = {'image_id': int(img_id), 'caption': sentence}
        sentence_list.append(item)


    print('*** compute scores ***')
    sen_json = save_sentence(sentence_list, 1, sen_path)
    results = coco_metrics(test_cap, sen_json)

    save_metrics(results, 1, metrics_path)
    print('*** complete prediction ***')