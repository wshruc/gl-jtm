import torch
from data_preprocess import DataManager
from torch.autograd import Variable
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import random


def load_model():
    save_best_model = f'./save_models/sq_model.pt'
    print('Load best model', save_best_model)
    with open(save_best_model, 'rb') as infile:
        model = torch.load(infile)
    return model


def idx2word(self, id_sentence, id_type  = 'word'):
    dic = {idx: word for word, idx in self.word_dic.items()}
    word_sentence = []
    for idx in id_sentence:
        if idx == 0:
            continue
        word_sentence.append(dic[idx])
    return word_sentence


def padding_data(dm, q_list_id, w_list_id, r_list_id, maxlen):
    q = dm.pad_obj(maxlen, q_list_id)
    p_word = dm.pad_obj(maxlen, w_list_id)
    r_word = dm.pad_obj(1, r_list_id)
    return q, p_word,r_word


if __name__ == "__main__":
    dm = DataManager('sq')
    question = "where is the passaic river located"
    pos_id = 379
    MAX_LEN = 16
    q_list = question.strip().split(' ')
    print(q_list)
    rel_list, word_list = dm.split_relation(379)
    print(rel_list)
    print(word_list)
    q_list_id = list(
        map(lambda x: dm.word_dic[x] if x in dm.word_dic else dm.word_dic['<unk>'], q_list))
    w_list_id = list(
        map(lambda x: dm.word_dic[x] if x in dm.word_dic else dm.word_dic['<unk>'], word_list))

    r_list_id = list(
        map(lambda x: dm.rel_dic[x] if x in dm.rel_dic else dm.rel_dic['<unk>'], rel_list))

    # print(q_list_id)
    # print(w_list_id)
    # print(r_list_id)

    # relation_list_id  = w_list_id + r_list_id

    p_q, p_w,p_r = padding_data(dm, q_list_id, w_list_id, r_list_id,  MAX_LEN)
    print(p_q)
    print(p_w)
    print(p_r)
    q = Variable(torch.LongTensor(p_q)).cuda()
    p_words = Variable(torch.LongTensor(p_w)).cuda()
    p_rel =Variable(torch.LongTensor(p_r)).cuda()
    q = q.unsqueeze(0)
    p_words = p_words.unsqueeze(0)
    p_rel = p_rel.unsqueeze(0)


    q_mask = []
    for word in q:
        q_mask.append([1 if i != 0 else 0 for i in word])
    q_mask = Variable(torch.LongTensor(q_mask)).cuda()


    model = load_model()
    model = model.eval().cuda()
    _,_, w_weight = model(q, p_words, p_rel, q_mask)
    # print("q_weight size:", q_weight.size())
    print("w_weight size:", w_weight.size())
    # q_weight = q_weight.squeeze(0)  #[20, 22]
    w_weight = w_weight.squeeze(0) #[22, 20]
    # q_weight = q_weight[:len(q_list), :len(word_list)+1]
    w_weight = w_weight[:len(word_list)+1,:len(q_list)]

    # print("q_weight size:", q_weight.size())
    print("w_weight size:", w_weight.size())

    # plt.figure(dpi=300, figsize=(30, 30))
    # q_ax = plt.subplot(111)
    # q_ax.matshow(q_weight.detach().to('cpu').numpy())
    # plt.xticks([])
    # plt.yticks([])
    # q_ax.set_xticks([i for i in range(len(q_list))])
    # q_ax.set_yticks([i for i in range(len(r_list))])
    # q_ax.set_xticklabels(q_list, rotation=30, fontsize=25)
    # q_ax.set_yticklabels(r_list, rotation=30, fontsize=25)

    # plt.savefig("./rq_weight_attn.pdf")

    r_ax = plt.subplot(111)
    r_ax.matshow(w_weight.detach().to('cpu').numpy())
    plt.xticks([])
    plt.yticks([])
    # r_ax.set_xticks([i for i in range(len(r_list))])
    # r_ax.set_yticks([i for i in range(len(r_list))])
    # r_ax.set_xticklabels(r_list, rotation=45, fontsize=25)
    # r_ax.set_yticklabels(r_list, rotation=45, fontsize=25)
    plt.savefig("./w_weight_attn.pdf")
