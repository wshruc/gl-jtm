import os
import sys

sys.path.append('../')
from tools import log
import logging

if __name__ == '__main__':

    gold_mid_list = []
    gold_ent_list = []
    dect_ent_list = []
    gold_rel_list = []
    pred_rel_list = []

    # right = 0
    # SimpleQuestions
    data_lines = open('../dataset/sq_relations/test_data.txt', 'r', encoding = 'utf-8').readlines()
    relstion_lines = open("../results/sq_result/relation_detection_result.txt", 'r', encoding = 'utf-8').readlines()
    entity_lines = open("../results/sq_result/entity_detection_result.txt", 'r', encoding = 'utf-8').readlines()
    data_for_test = '../kbqa/data/sq_data_test.txt'
    data_test = open(data_for_test, 'w', encoding = 'utf-8')

    for line in entity_lines:
        entity = line.strip()
        dect_ent_list.append(entity)

    for line in relstion_lines:
        items = line.split('\t')
        gold_list = items[1].split('/')
        gold_list = gold_list[1:]
        gold = '.'.join(gold_list).strip()
        pred_list = items[2].split('/')
        pred_list = pred_list[1:]
        pred = '.'.join(pred_list).strip()
        gold_rel_list.append(gold)
        pred_rel_list.append(pred)

    for line in data_lines:
        items = line.split('||')
        gold_mid_list.append(items[3])
        gold_ent_list.append(items[6])

    # for i in range(len(dect_ent_list)):
    #     if dect_ent_list[i] == gold_ent_list[i]:
    #         right +=1
    # print(right)


    count = 0
    for idx in range(len(dect_ent_list)):
        assert len(dect_ent_list) == len(dect_ent_list) == len(gold_rel_list) == len(pred_rel_list)
        line_str = dect_ent_list[idx].strip() + '\t' + gold_mid_list[idx].strip() + '\t' + gold_rel_list[
            idx].strip() + '\t' + pred_rel_list[idx].strip() + '\n'

        data_test.write(line_str)
        count += 1

    if count % 1000 == 0:
        logging.info('handle {} lines...'.format(count))
