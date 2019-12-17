'''save information function'''
'''一些保存信息的函数'''
import os
import json
import torch
import pandas as pd


def create_file(model_name, version, cfg):
    ''' create save training information file'''
    '''创建保存训练结果的文件'''

    # create directory (创建保存文件夹)
    if not os.path.exists(cfg.log.format(model_name, version)):
        os.makedirs(cfg.log.format(model_name, version))
    if not os.path.exists(cfg.checkpoint.format(model_name, version)):
        os.makedirs(cfg.checkpoint.format(model_name, version))

    # supplement path (补充路径）
    loss = cfg.loss.format(model_name, version)
    metrics = cfg.metrics.format(model_name, version)
    sentence = cfg.sentence.format(model_name, version)
    best_model = cfg.best_model.format(model_name, version)

    return loss, metrics, sentence, best_model


def create_result(model_name, version, cfg):
    ''' create save result information file'''
    '''创建保存预测结果的文件'''

    # create directory (创建保存文件夹)
    if not os.path.exists(cfg.eval_log.format(model_name, version)):
        os.makedirs(cfg.eval_log.format(model_name, version))

    # supplement path (补充路径）
    metrics = cfg.eval_metrics.format(model_name, version)
    sentence = cfg.eval_sen.format(model_name, version)

    return metrics, sentence
def save_loss(epoch_loss, epoch, save_path):
    '''save loss information and output'''
    '''保存loss信息并打印'''
    epoch_loss = round(epoch_loss, 3)
    if epoch == 1:
        loss_file = pd.DataFrame(columns=['epoch', 'loss'])
        loss_file.to_csv(save_path, index=0)
    loss_file = pd.read_csv(save_path)
    loss_file = loss_file.append({'epoch': epoch, 'loss': epoch_loss}, ignore_index=True)
    loss_file.to_csv(save_path, index=0)

    print("*** Epoch:{},Loss:{} ***".format(epoch, epoch_loss))

def save_sentence(sentece, epoch, save_folder):
    '''save epoch sentences in val set(json format)'''
    '''保存每一轮val集上生成的句子(json格式）'''
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    save_path = os.path.join(save_folder, '{}.json'.format(epoch))
    with open(save_path, 'w') as f:
        json.dump(sentece, f)

    return save_path

def save_metrics(results, epoch, save_path):
    '''save epoch scores'''
    '''保存每一轮的分数'''
    if epoch == 1:
        metrics_list = ['epoch', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
        metrics_file = pd.DataFrame(columns=metrics_list)
        metrics_file.to_csv(save_path, index=0)

    metrics_file = pd.read_csv(save_path)
    score_dict = {'epoch': epoch}
    for metric, score in results:
        score_dict[metric] = round(score, 3)
    metrics_file = metrics_file.append(score_dict, ignore_index=True)
    metrics_file.to_csv(save_path, index=0)

    return score_dict

def save_best_model(model, optimizer, epoch, best_score_dict, best_epoch, best_model_path):
    ''' save the best model'''
    '''保存最好的模型'''
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_score_dict': best_score_dict,
        'best_epoch': best_epoch,
    }, best_model_path)