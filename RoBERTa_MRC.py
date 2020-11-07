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
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2][4:]
from typing import Any

from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import pickle
from torch.utils.data import DataLoader, Dataset
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter


class MyDataset(Dataset):
    def __init__(self, data, max_enc_len):
        self.data = data
        self.max_enc_len = max_enc_len
        self.SEG_Q = 0
        self.SEG_P = 1
        self.ID_PAD = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        """
        query context表示问题和响应的参考文本, distance表示当前文本开头相比原始文本开头的偏移量
        start是当前文本属于非滑窗的开始位置, q_type是问题类型, answer_list则是对应的回答列表
        """
        query, context, distance, start, q_type, answer_list = item["query"], item["context"], item["distance"], \
                                                               item["start"], item["type"], item["answer"]
        # 首先按照字符编码query和context ==> 可以确保query + context + 三标记符长度小于等于max_enc_len
        query_tokens, context_tokens = [i for i in query], [i for i in context]
        c = ["[CLS]"] + query_tokens + ["[SEP]"] + context_tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(c)
        input_mask = [1.0] * len(input_ids)
        input_seg = [self.SEG_Q] * (len(query_tokens) + 2) + [self.SEG_P] * (len(input_ids) - 2 - len(query_tokens))
        extra = self.max_enc_len - len(input_ids)
        if extra > 0:
            input_ids += [self.ID_PAD] * extra
            input_mask += [0.0] * extra
            input_seg += [self.SEG_P] * extra
        # 其次, 编码seq_mask => 绑定start
        seq_mask = np.zeros(shape=(self.max_enc_len,))
        t_s = len(query_tokens) + 2 + start
        t_e = len(query_tokens) + 2 + len(context_tokens)
        assert t_e < self.max_enc_len
        for i in range(t_s, t_e):
            seq_mask[i] = 1.0
        # 最后编码label
        start_label, end_label = \
            np.zeros(shape=(self.max_enc_len,), dtype=np.int), np.zeros(shape=(self.max_enc_len,), dtype=np.int)
        for answer in answer_list:
            ans_s, ans_e = answer["ans_s"], answer["ans_e"]
            start_label[ans_s - distance + 2 + len(query_tokens)] = 1
            end_label[ans_e - distance + 2 + len(query_tokens)] = 1
        return {
            "input_ids": torch.tensor(input_ids).long(), "input_seg": torch.tensor(input_seg).long(),
            "input_mask": torch.tensor(input_mask).float(), "seq_mask": torch.tensor(seq_mask).float(),
            "start_label": torch.tensor(start_label).long(), "end_label": torch.tensor(end_label).long()
        }


class EvalDataset(object):
    def __init__(self, data, max_enc_len, is_test=False):
        self.data = data
        self.max_enc_len = max_enc_len
        self.SEG_Q = 0
        self.SEG_P = 1
        self.ID_PAD = 0
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        query_list = x["query"]
        input_ids_list, input_seg_list, input_mask_list = [], [], []
        type_list, start_list, end_list = [], [], []
        distance_list, context_list, c_start_list = [], [], []
        for item in query_list:
            query, context, distance, start, q_type = item["query"], item["context"], item["distance"], \
                                                      item["start"], item["type"]
            type_list.append(q_type)
            context_list.append(context)
            distance_list.append(distance)
            query_tokens, context_tokens = [i for i in query], [i for i in context]
            c = ["[CLS]"] + query_tokens + ["[SEP]"] + context_tokens + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(c)
            input_mask = [1.0] * len(input_ids)
            input_seg = [self.SEG_Q] * (len(query_tokens) + 2) + [self.SEG_P] * (len(input_ids) - 2 - len(query_tokens))
            extra = self.max_enc_len - len(input_ids)
            if extra > 0:
                input_ids += [self.ID_PAD] * extra
                input_mask += [0.0] * extra
                input_seg += [self.SEG_P] * extra
            input_ids_list.append(torch.tensor(input_ids).long()[None, :])
            input_seg_list.append(torch.tensor(input_seg).long()[None, :])
            input_mask_list.append(torch.tensor(input_mask).float()[None, :])
            t_s = len(query_tokens) + 2 + start
            t_e = len(query_tokens) + 2 + len(context_tokens)
            assert t_e < self.max_enc_len
            start_list.append(t_s)
            end_list.append(t_e)
            c_start_list.append(len(query_tokens) + 2)
        if len(input_ids_list) % args["batch_size"] == 0:
            chunk_num = len(input_ids_list) // args["batch_size"]
        else:
            chunk_num = (len(input_ids_list) // args["batch_size"]) + 1
        out = {
            "input_ids_chunks": torch.chunk(torch.cat(input_ids_list, dim=0), chunks=chunk_num, dim=0),
            "input_seg_chunks": torch.chunk(torch.cat(input_seg_list, dim=0), chunks=chunk_num, dim=0),
            "input_mask_chunks": torch.chunk(torch.cat(input_mask_list, dim=0), chunks=chunk_num, dim=0),
            "type_list": type_list, "start_list": start_list, "end_list": end_list,
            "c_start_list": c_start_list, "distance_list": distance_list,
            "context_list": context_list
        }
        if self.is_test:
            out["id"] = x["id"]
        else:
            out["answer"] = x["answer"]
        return out


class MyModel(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, pre_train_dir: str, dropout_rate: float):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=1024),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout_rate)
        )
        self.start_layer = torch.nn.Linear(in_features=1024, out_features=2)
        self.end_layer = torch.nn.Linear(in_features=1024, out_features=2)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        self.alpha = args["alpha"]
        self.gamma = args["gamma"]
        self.epsilon = 1e-6

    def forward(self, input_ids, input_mask, input_seg, start_label=None, end_label=None, seq_mask=None):
        bsz, seq_len = input_ids.size()[0], input_ids.size()[1]
        encoder_rep = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)[0]  # (bsz, seq, dim)
        encoder_rep = self.encoder_linear(encoder_rep)
        start_logits = self.start_layer(encoder_rep)  # (bsz, seq, 2)
        end_logits = self.end_layer(encoder_rep)  # (bsz, seq, 2)
        # 将两个span组合 => (bsz, seq, seq)
        start_prob_seq = torch.nn.functional.softmax(start_logits, dim=-1)  # (bsz, seq, 2)
        end_prob_seq = torch.nn.functional.softmax(end_logits, dim=-1)  # (bsz, seq, 2)
        if start_label is None or end_label is None or seq_mask is None:
            return start_prob_seq, end_prob_seq
        else:
            # 计算start和end的loss
            start_label, end_label = start_label.view(size=(-1,)), end_label.view(size=(-1,))
            start_loss = self.loss_func(input=start_logits.view(size=(-1, 2)), target=start_label)
            end_loss = self.loss_func(input=end_logits.view(size=(-1, 2)), target=end_label)
            # 加入focal loss实现 => 效果更差了, emmmm
            """start_prob_seq, end_prob_seq = start_prob_seq.view(size=(-1, 2)), end_prob_seq.view(size=(-1, 2))
            start_loss = (1 - start_label) * ((1 - start_prob_seq[:, 0]) ** self.gamma) * start_loss + \
                         self.alpha * start_label * ((1 - start_prob_seq[:, 1]) ** self.gamma) * start_loss
            end_loss = (1 - end_label) * ((1 - end_prob_seq[:, 0]) ** self.gamma) * end_loss + \
                       self.alpha * end_label * ((1 - end_prob_seq[:, 1]) ** self.gamma) * end_loss"""
            sum_loss = start_loss + end_loss
            sum_loss *= seq_mask.view(size=(-1,))

            avg_se_loss = self.alpha * torch.sum(sum_loss) / (torch.nonzero(seq_mask, as_tuple=False).size()[0])
            # avg_se_loss = torch.sum(sum_loss) / bsz
            return avg_se_loss[None].repeat(bsz)


class WarmUp_LinearDecay:
    def __init__(self, optimizer: optim.AdamW, init_rate, warm_up_steps, decay_steps, min_lr_rate):
        self.optimizer = optimizer
        self.init_rate = init_rate
        self.warm_up_steps = warm_up_steps
        self.decay_steps = decay_steps
        self.min_lr_rate = min_lr_rate
        self.optimizer_step = 0

    def step(self):
        self.optimizer_step += 1
        if self.optimizer_step <= self.warm_up_steps:
            rate = (self.optimizer_step / self.warm_up_steps) * self.init_rate
        elif self.warm_up_steps < self.optimizer_step <= (self.warm_up_steps + self.decay_steps):
            rate = (1.0 - ((self.optimizer_step - self.warm_up_steps) / self.decay_steps)) * self.init_rate
        else:
            rate = self.min_lr_rate
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()


class Main(object):
    def __init__(self, train_loader, valid_loader, test_flag=False, test_loader=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model = MyModel(pre_train_dir=args["pre_train_dir"], dropout_rate=args["dropout"])

        if test_flag:
            self.model.load_state_dict(torch.load(args["save_path"], map_location=device), strict=False)
        else:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': args["weight_decay"]},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]

            self.optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=args["init_lr"])
            self.schedule = WarmUp_LinearDecay(
                optimizer=self.optimizer, init_rate=args["init_lr"], warm_up_steps=args["warm_up_steps"],
                decay_steps=args["lr_decay_steps"], min_lr_rate=args["min_lr_rate"]
            )
        self.model.to(device=device)
        if args["is_train"]:
            self.model = nn.parallel.DistributedDataParallel(module=self.model, dim=0, find_unused_parameters=True)

    def train(self):
        best_f1 = 0
        self.model.train()
        steps = 0
        while True:
            for item in self.train_loader:
                input_ids, input_mask, input_seg, seq_mask, start_label, end_label = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["seq_mask"], \
                    item["start_label"], item["end_label"]
                self.optimizer.zero_grad()
                loss = self.model(
                    input_ids=input_ids.to(device), input_mask=input_mask.to(device),
                    input_seg=input_seg.to(device), seq_mask=seq_mask.to(device),
                    start_label=start_label.to(device), end_label=end_label.to(device)
                )
                loss = loss.float().mean().type_as(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=args["clip_norm"])
                self.schedule.step()
                steps += 1
                writer.add_scalar("loss", loss.item(), global_step=steps)
                if steps % args["eval_interval"] == 0:
                    p, r, f1 = self.eval()
                    writer.add_scalar("valid/F1", f1, global_step=steps)
                    writer.add_scalar("valid/P", p, global_step=steps)
                    writer.add_scalar("valid/R", r, global_step=steps)
                    if f1 > best_f1:
                        best_f1 = f1
                        torch.save(self.model.module.state_dict(), f=args["save_path"])
                if steps >= args["max_steps"]:
                    break
            if steps >= args["max_steps"]:
                break
        writer.flush()
        writer.close()

    def eval(self):
        self.model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for item in self.valid_loader:
                input_ids_chunks, input_seg_chunks, input_mask_chunks, type_list = \
                    item["input_ids_chunks"], item["input_seg_chunks"], item["input_mask_chunks"], item["type_list"]
                start_list, end_list, c_start_list, answer_list, distance_list, context_list = \
                    item["start_list"], item["end_list"], item["c_start_list"], item["answer"], \
                    item["distance_list"], item["context_list"]
                y_true.append({"answer": item["answer"]})
                s_seq_list, e_seq_list = [], []
                for i in range(len(input_ids_chunks)):
                    s_seq, e_seq = self.model(
                        input_ids=input_ids_chunks[i].to(device), input_mask=input_mask_chunks[i].to(device),
                        input_seg=input_seg_chunks[i].to(device)
                    )
                    s_seq_list.extend(s_seq.cpu().numpy())
                    e_seq_list.extend(e_seq.cpu().numpy())
                tmp = []
                for i in range(len(s_seq_list)):
                    tmp.extend(self.sequence_dec(
                        s_seq=s_seq_list[i], e_seq=e_seq_list[i],
                        q_type=type_list[i], start=start_list[i], end=end_list[i],
                        c_start=c_start_list[i], distance=distance_list[i],
                        context=context_list[i]
                    ))
                y_pred.append({"answer": tmp})
        self.model.train()
        return self.calculate_strict_f1(y_pred=y_pred, y_true=y_true)

    def test(self):
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for item in self.test_loader:
                input_ids_chunks, input_seg_chunks, input_mask_chunks, type_list = \
                    item["input_ids_chunks"], item["input_seg_chunks"], item["input_mask_chunks"], item["type_list"]
                start_list, end_list, c_start_list, text_id, distance_list, context_list = \
                    item["start_list"], item["end_list"], item["c_start_list"], item["id"], \
                    item["distance_list"], item["context_list"]
                s_seq_list, e_seq_list = [], []
                for i in range(len(input_ids_chunks)):
                    s_seq, e_seq = self.model(
                        input_ids=input_ids_chunks[i].to(device), input_mask=input_mask_chunks[i].to(device),
                        input_seg=input_seg_chunks[i].to(device)
                    )
                    s_seq_list.extend(s_seq.cpu().numpy())
                    e_seq_list.extend(e_seq.cpu().numpy())
                tmp = []
                for i in range(len(s_seq_list)):
                    tmp.extend(self.sequence_dec(
                        s_seq=s_seq_list[i], e_seq=e_seq_list[i],
                        q_type=type_list[i], start=start_list[i], end=end_list[i],
                        c_start=c_start_list[i], distance=distance_list[i],
                        context=context_list[i]
                    ))
                y_pred.append((text_id, tmp))
        str_format = "T{number}\t{type} {start} {end}\t{label}\n"
        for text_id, answer_list in y_pred:
            with open("Submit/%s.ann" % text_id, "w", encoding="UTF-8") as f:
                for i, item in enumerate(answer_list):
                    f.write(str_format.format(number=i + 1, type=item["type"], start=item["start"],
                                              end=item["end"], label=item["label"]))
                f.flush()

    @staticmethod
    def sequence_dec(s_seq, e_seq, q_type, start, end, c_start, distance, context):
        ans_index = []
        i_start, i_end = [], []
        for i in range(start, end + 1):
            if s_seq[i][1] > args["threshold"]:
                i_start.append(i)
            if e_seq[i][1] > args["threshold"]:
                i_end.append(i)
        # 然后遍历i_end
        cur_end = -1
        for e in i_end:
            s = []
            for i in i_start:
                if e >= i >= cur_end and (e - i) <= max_dec_len_map[q_type]:
                    s.append(i)
            max_s = 0.0
            t = None
            for i in s:
                if s_seq[i][1] + e_seq[e][1] > max_s:
                    t = (i, e)
                    max_s = s_seq[i][1] + e_seq[e][1]
            cur_end = e
            if t is not None:
                ans_index.append(t)
        out = []
        for item in ans_index:
            out.append({"type": q_type, "start": distance + item[0] - c_start, "end": distance + item[1] + 1 - c_start,
                        "label": context[item[0]-c_start: item[1]-c_start+1]})
        return out

    @staticmethod
    def calculate_strict_f1(y_pred, y_true):
        # y_pred列表形式, 每一项为一字典 => {"answer": ["type", "start", "end"]}
        # y_true列表形式, 每一项为一字典 => {"answer": ["type", "ans_s", "ans_e", "label"]}
        sum_pred, sum_true = 0, 0
        exact_match = 0
        for i in range(len(y_pred)):
            p_answer, t_answer = y_pred[i]["answer"], y_true[i]["answer"]
            sum_pred += len(p_answer)
            sum_true += len(t_answer)
            for item in p_answer:
                for jtem in t_answer:
                    if item["type"] == jtem["type"] and item["start"] == jtem["ans_s"] and item["end"] == jtem["ans_e"]:
                        exact_match += 1
        precision, recall = exact_match / sum_pred, exact_match / sum_true
        return precision, recall, (2 * precision * recall) / (precision + recall + 1e-5)


if __name__ == "__main__":
    device = "cuda"
    args = {
        "init_lr": 2e-5,
        "batch_size": 24,
        "weight_decay": 0.01,
        "warm_up_steps": 700,
        "lr_decay_steps": 3000,
        "max_steps": 4000,
        "min_lr_rate": 1e-8,
        "eval_interval": 200,
        "save_path": "ModelStorage/best_score.pth",
        "pre_train_dir": "/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/",
        "clip_norm": 0.25,
        "dimension": 1024,
        "max_enc_len": 512,
        "alpha": 1.0,
        "gamma": 2.0,
        "dropout": 0.1,
        "threshold": 0.45  # 当前best_score对应的模型阈值为0.45, 得分为0.7680
    }

    with open("process.pkl", "rb") as f:
        x = pickle.load(f)
    query_map, max_dec_len_map = x["query_map"], x["max_dec_len_map"]
    tokenizer = BertTokenizer(vocab_file="/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/vocab.txt")

    if sys.argv[1] == "train":
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1, init_method='tcp://localhost:6666')
        args["is_train"] = True
        writer = SummaryWriter(logdir="RunLog/%s" % sys.argv[3])
        train_dataset = MyDataset(data=x["train_items"], max_enc_len=args["max_enc_len"])
        train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4)

        valid_loader = EvalDataset(data=x["valid_items"], max_enc_len=args["max_enc_len"])

        m = Main(train_loader, valid_loader)
        m.train()
    else:
        args["is_train"] = False
        writer = None
        test_loader = EvalDataset(data=x["test_items"], max_enc_len=args["max_enc_len"], is_test=True)
        m = Main(None, None, test_flag=True, test_loader=test_loader)
        m.test()



