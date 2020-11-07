#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time: 2020-09-28
    @Author: menghuanlater
    @Software: Pycharm 2019.2
    @Usage:
-----------------------------
    Description:
-----------------------------
"""
import pickle
import os
from random import shuffle
import re

output = {
    "query_map": {
        "DRUG": "找出所有的中药:指在中医理论指导下，用于预防、治疗、诊断疾病并具有康复与保健作用的物质，"
                "主要来源是天然药及其加工品，包括植物药、动物药、矿物药及部分化学、生物制品类药物，如六味地黄丸、逍遥散等",
        "DRUG_INGREDIENT": "找出所有的中药组成成分:指中药复方中所含有的所有与该复方临床应用目的密切相关的药理活性成分，"
                           "如当归、人参、枸杞等",
        "DISEASE": "找出所有的疾病:指人体在一定原因的损害性作用下，因自稳调节紊乱而发生的异常生命活动过程，是特定的异常病理情形，"
                   "而且会影响生物体的部分或是所有器官，伴随着特定的症状及医学征象，如高血压、心绞痛、糖尿病等",
        "SYMPTOM": "找出所有的症状:指疾病过程中机体内的一系列机能、代谢和形态结构异常变化所引起的病人主观上的异常感觉或某些客观病态改变，"
                   "如头晕、心悸、小腹胀痛等",
        "SYNDROME": "找出所有的证候:表示一系列有相互关联的症状总称，如血瘀、气滞、气血不足、气血两虚等",
        "DISEASE_GROUP": "找出所有的疾病分组:指疾病涉及有人体组织部位的疾病名称的统称概念，非某项具体医学疾病，如肾病、肝病、肺病等",
        "FOOD": "找出所有的食物:指能够满足机体正常生理和生化能量需求，并能延续正常寿命的物质。对人体而言，其能够满足人的正常生活活动"
                "需求并利于寿命延长，如苹果、茶、木耳、萝卜等",
        "FOOD_GROUP": "找出所有的食物分组:指中医饮食养生中，将食物分为寒热温凉四性，同时中医药禁忌中对于具有某类共同属性食物的统称，"
                      "如油腻食物、辛辣食物、凉性食物等",
        "PERSON_GROUP": "找出所有的人群:指中医药的适用及禁忌范围内相关特定人群，如孕妇、经期妇女、儿童、青春期少女等",
        "DRUG_GROUP": "找出所有的药品分组:指具有某一类共同属性的药品类统称概念，非某项具体药品名，如止咳药、退烧药等",
        "DRUG_DOSAGE": "找出所有的药物剂型:药物在供给临床使用前，均必须制成适合于医疗和预防应用的形式，成为药物剂型，如浓缩丸、水蜜丸、糖衣片等",
        "DRUG_TASTE": "找出所有的药品气味:药品的性质和气味，如味甘、酸涩、气凉等",
        "DRUG_EFFICACY": "找出所有的中药功效: 药品的主治功能和效果的统称，如滋阴补肾、去瘀生新、活血化瘀等"
    },
    "train_items": [],
    "valid_items": [],
    "test_items": [],
    "max_dec_len_map": dict(),
    "answer_category_distribution": dict()
}


def construct_query_context(q_type, context):
    query = output["query_map"][q_type]
    r = []
    if len(context) <= 509 - len(query):
        r.append({"query": query, "context": context, "distance": 0, "start": 0, "type": q_type})
    else:
        d, s = 0, 0
        while True:
            if len(context[d:]) <= 509 - len(query):
                r.append({"query": query, "context": context[d:], "distance": d, "start": s, "type": q_type})
                break
            else:
                x = 0
                while context[d + 509 - len(query) - 1 - x] not in [" ", "\t", "，", "。"]:
                    x += 1
                r.append({"query": query, "context": context[d:(d + 509 - len(query) - x)], "distance": d, "start": s, "type": q_type})
                s = 509 - len(query) - x - 200
                d += 200
    return r


train_txt = [i for i in os.listdir("DataSet/train") if ".txt" in i]
shuffle(train_txt)
train_txt, valid_txt = train_txt[50:], train_txt[:50]
test_txt = [i for i in os.listdir("DataSet/test")]

for txt in train_txt:
    digit = int(txt[:-4])
    context = open("DataSet/train/%s" % txt, "r", encoding="UTF-8").readline()
    ann = open("DataSet/train/%d.ann" % digit, "r", encoding="UTF-8").readlines()
    src = []
    for item in ann:
        t = re.split("\s", item)
        q_type, s, e, label = t[1], t[2], t[3], t[4]
        src.append({"type": q_type, "start": int(s), "end": int(e) - 1, "label": label})
        if q_type not in output["answer_category_distribution"].keys():
            output["answer_category_distribution"][q_type] = 1
        else:
            output["answer_category_distribution"][q_type] += 1
        if q_type not in output["max_dec_len_map"].keys():
            output["max_dec_len_map"][q_type] = len(label)
        else:
            if len(label) > output["max_dec_len_map"][q_type]:
                output["max_dec_len_map"][q_type] = len(label)
    for key in output["query_map"].keys():
        q_c_pairs = construct_query_context(key, context)
        for item in q_c_pairs:
            item["answer"] = []
            i_s, i_e = item["distance"] + item["start"], item["distance"] + len(item["context"]) - 1
            for jtem in src:
                if jtem["type"] == key and jtem["start"] >= i_s and jtem["end"] <= i_e:
                    item["answer"].append({"ans_s": jtem["start"], "ans_e": jtem["end"], "ans_label": jtem["label"]})
        output["train_items"].extend(q_c_pairs)

for i in range(3):
    shuffle(output["train_items"])

for txt in valid_txt:
    digit = int(txt[:-4])
    context = open("DataSet/train/%s" % txt, "r", encoding="UTF-8").readline()
    ann = open("DataSet/train/%d.ann" % digit, "r", encoding="UTF-8").readlines()
    t = {"query": [], "answer": []}
    output["valid_items"].append(t)
    for item in ann:
        m = re.split("\s", item)
        q_type, s, e, label = m[1], m[2], m[3], m[4]
        t["answer"].append({"type": q_type, "ans_s": int(s), "ans_e": int(e), "label": label})
    for key in output["query_map"].keys():
        q_c_pairs = construct_query_context(key, context)
        t["query"].extend(q_c_pairs)

for txt in test_txt:
    digit = int(txt[:-4])
    context = open("DataSet/test/%s" % txt, "r", encoding="UTF-8").readline()
    t = {"id": str(digit), "query": []}
    output["test_items"].append(t)
    for key in output["query_map"].keys():
        q_c_pairs = construct_query_context(key, context)
        t["query"].extend(q_c_pairs)

print(len(output["train_items"]))
# print(output["max_dec_len_map"])
print(output["answer_category_distribution"])
with open("process.pkl", "wb") as f:
   pickle.dump(output, f)
