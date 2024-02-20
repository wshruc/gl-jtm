import os
import sys

sys.path.append('../')
from tools import log
import logging
import string
import unicodedata
"""
Construct the test data for KBQA system according
to the result of entity detection and relation detection
"""

if __name__ == '__main__':
    dect_ent_list = []

    pred_rel_list = []
    gold_rel_list = []


    question_list = []
    answer_list = []
    gold_mention_list = []
    gold_entity_list = []
    gold_mid_list = []
    error_list = []
    # WebQuestions
    wq_test_lines = open('../dataset/webqsp_relations/test_data.txt', 'r', encoding = 'utf-8').readlines()
    relstion_lines = open("../results/wq_result/relation_detection_result.txt", 'r', encoding = 'utf-8').readlines()
    entity_lines = open("../results/wq_result/entity_detection_result.txt", 'r', encoding = 'utf-8').readlines()
    data_for_test = '../kbqa/data/wq_data_test.txt'
    data_test = open(data_for_test, 'w', encoding = 'utf-8')

    for line in entity_lines:
        entity = line.strip()
        dect_ent_list.append(entity.strip())


    for line in relstion_lines:
        items = line.split('\t')
        rels = items[2].strip()
        pred_rel_list.append(rels)
        grels = items[1].strip()
        gold_rel_list.append(grels)

    for line in wq_test_lines:
        items = line.split('||')
        question_list.append(items[1].strip())
        answer_list.append(items[7].strip())
        gold_mention_list.append(items[3].strip())
        gold_entity_list.append(items[4].strip())
        gold_mid_list.append(items[5].strip())


    count = 0
    for idx in range(len(question_list)):
        count += 1
        print(count)
        ans = answer_list[idx].strip()
        if ans == "":
            continue
        # if "g." not in ans:
        #     continue
        if "m." not in ans:
            continue

        line_str = question_list[idx].strip() + '||' + dect_ent_list[idx].strip() + \
                   "||" + pred_rel_list[idx].strip() + "||" + gold_mention_list[idx].strip() + \
                   "||" + gold_entity_list[idx].strip() + "||" + gold_mid_list[idx].strip() + \
                   "||" + gold_rel_list[idx].strip() + "||" + ans + '\n'

        data_test.write(line_str)
