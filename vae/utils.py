import random
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from squad_utils import (convert_examples_to_features_answer_id,
                         convert_examples_to_harv_features,
                         read_squad_examples)


def get_squad_data_loader(tokenizer, file, shuffle, args):
    cache_file = f"{file}.cache"
    if os.path.exists(cache_file) and args.debug:
        loaddata = torch.load(cache_file)
        data_loader, examples, features = loaddata["data_loader"], loaddata["examples"],loaddata["features"]
        return data_loader, examples, features
    examples = read_squad_examples(file, is_training=True, debug=args.debug)
    features = convert_examples_to_features_answer_id(examples,
                                                      tokenizer=tokenizer,
                                                      max_seq_length=args.max_c_len,
                                                      max_query_length=args.max_q_len,
                                                      max_ans_length=args.max_q_len,
                                                      doc_stride=128,
                                                      is_training=True)
    # 所有context的id, all_q_ids是问题的id, all_a_ids是问题的id, all_tag_ids: 答案在上下文中的编码： 类似BIO的策略，答案开始的位置设为1，其它答案位置设为2，剩余所有上下文为0
    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    all_q_ids = torch.tensor([f.q_ids for f in features], dtype=torch.long)
    all_tag_ids = torch.tensor([f.tag_ids for f in features], dtype=torch.long)
    all_a_ids = (all_tag_ids != 0).long()
    all_start_positions = torch.tensor([f.noq_start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.noq_end_position for f in features], dtype=torch.long)

    all_data = TensorDataset(all_c_ids, all_q_ids, all_a_ids, all_start_positions, all_end_positions)
    data_loader = DataLoader(all_data, args.batch_size, shuffle=shuffle)
    # 返回dataloder，样本原始数据，和特征数据（变成id的数据）
    if args.debug:
        torch.save({"data_loader":data_loader, "examples":examples, "features":features}, cache_file)
    return data_loader, examples, features

def get_harv_data_loader(tokenizer, file, shuffle, ratio, args):
    examples = read_squad_examples(file, is_training=True, debug=args.debug)
    random.shuffle(examples)
    num_ex = int(len(examples) * ratio)
    examples = examples[:num_ex]
    features = convert_examples_to_harv_features(examples,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=args.max_c_len,
                                                 max_query_length=args.max_q_len,
                                                 doc_stride=128,
                                                 is_training=True)
    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_c_ids)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=args.batch_size)

    return features, dataloader

def batch_to_device(batch, device):
    batch = (b.to(device) for b in batch)
    c_ids, q_ids, a_ids, start_positions, end_positions = batch

    c_len = torch.sum(torch.sign(c_ids), 1)
    max_c_len = torch.max(c_len)
    c_ids = c_ids[:, :max_c_len]
    a_ids = a_ids[:, :max_c_len]

    q_len = torch.sum(torch.sign(q_ids), 1)
    max_q_len = torch.max(q_len)
    q_ids = q_ids[:, :max_q_len]

    return c_ids, q_ids, a_ids, start_positions, end_positions
