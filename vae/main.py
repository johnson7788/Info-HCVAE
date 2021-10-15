import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import BertTokenizer

from eval import eval_vae
from trainer import VAETrainer
from utils import batch_to_device, get_harv_data_loader, get_squad_data_loader


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    train_loader, _, _ = get_squad_data_loader(tokenizer, args.train_dir,
                                         shuffle=True, args=args)
    eval_data = get_squad_data_loader(tokenizer, args.dev_dir,
                                      shuffle=False, args=args)
    #当前训练的设备
    args.device = torch.cuda.current_device()
    # 加载模型
    trainer = VAETrainer(args)

    loss_log1 = tqdm(total=0, bar_format='{desc}', position=2)
    loss_log2 = tqdm(total=0, bar_format='{desc}', position=3)
    eval_log = tqdm(total=0, bar_format='{desc}', position=5)
    best_eval_log = tqdm(total=0, bar_format='{desc}', position=6)

    print("MODEL DIR: " + args.model_dir)

    best_bleu, best_em, best_f1 = 0.0, 0.0, 0.0
    for epoch in trange(int(args.epochs), desc="Epoch", position=0):
        for batch in tqdm(train_loader, desc="Train iter", leave=False, position=1):
            c_ids, q_ids, a_ids, start_positions, end_positions \
            = batch_to_device(batch, args.device)
            trainer.train(c_ids, q_ids, a_ids, start_positions, end_positions)
            
            str1 = 'Q REC : {:06.4f} A REC : {:06.4f}'
            str2 = 'ZQ KL : {:06.4f} ZA KL : {:06.4f} INFO : {:06.4f}'
            str1 = str1.format(float(trainer.loss_q_rec), float(trainer.loss_a_rec))
            str2 = str2.format(float(trainer.loss_zq_kl), float(trainer.loss_za_kl), float(trainer.loss_info))
            loss_log1.set_description_str(str1)
            loss_log2.set_description_str(str2)

        if epoch > 10:
            metric_dict, bleu, _ = eval_vae(epoch, args, trainer, eval_data)
            f1 = metric_dict["f1"]
            em = metric_dict["exact_match"]
            bleu = bleu * 100
            _str = '{}-th Epochs BLEU : {:02.2f} EM : {:02.2f} F1 : {:02.2f}'
            _str = _str.format(epoch, bleu, em, f1)
            eval_log.set_description_str(_str)
            if em > best_em:
                best_em = em
            if f1 > best_f1:
                best_f1 = f1
                trainer.save(os.path.join(args.model_dir, "best_f1_model.pt"))
            if bleu > best_bleu:
                best_bleu = bleu
                trainer.save(os.path.join(args.model_dir, "best_bleu_model.pt"))

            _str = 'BEST BLEU : {:02.2f} EM : {:02.2f} F1 : {:02.2f}'
            _str = _str.format(best_bleu, best_em, best_f1)
            best_eval_log.set_description_str(_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1004, type=int, help='随机数种子')
    parser.add_argument('--debug', dest='debug', action='store_true', help='是否debug,会加载很少的训练数据进行debug')
    parser.add_argument('--train_dir', default='../data/squad/train-v1.1.json', help='训练数据')
    parser.add_argument('--dev_dir', default='../data/squad/my_dev.json', help='评估数据')
    
    parser.add_argument("--max_c_len", default=384, type=int, help="最大上下文长度")
    parser.add_argument("--max_q_len", default=64, type=int, help="最大查询长度")

    parser.add_argument("--model_dir", default="../save/vae-checkpoint", type=str,help='模型保存位置')
    parser.add_argument("--epochs", default=20, type=int,help='训练多少个epoch')
    parser.add_argument("--lr", default=1e-3, type=float, help="学习率")
    parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--clip", default=5.0, type=float, help="max grad norm")

    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,help='使用的预训练模型')
    parser.add_argument('--enc_nhidden', type=int, default=300, help='论文图1中所有编码器的隐藏层维度')
    parser.add_argument('--enc_nlayers', type=int, default=1,help='论文图1中的编码器的层数')
    parser.add_argument('--enc_dropout', type=float, default=0.2,help='编码器的dropout')
    parser.add_argument('--dec_a_nhidden', type=int, default=300, help='论文图1(b)中答案解码器的隐藏层维度')
    parser.add_argument('--dec_a_nlayers', type=int, default=1,help='论文图1(b)答案解码器层数')
    parser.add_argument('--dec_a_dropout', type=float, default=0.2,help='答案解码器dropout')
    parser.add_argument('--dec_q_nhidden', type=int, default=900,help='问题解码器的隐藏层维度')
    parser.add_argument('--dec_q_nlayers', type=int, default=2,help='论文图1（c)问题解码器的隐藏层层数')
    parser.add_argument('--dec_q_dropout', type=float, default=0.3)
    parser.add_argument('--nzqdim', type=int, default=50)
    parser.add_argument('--nza', type=int, default=20)
    parser.add_argument('--nzadim', type=int, default=10)
    parser.add_argument('--lambda_kl', type=float, default=0.1)
    parser.add_argument('--lambda_info', type=float, default=1.0,help='论文中最终损失公式中互信息损失的权重，影响问题和答案的一致性')

    args = parser.parse_args()

    if args.debug:
        args.model_dir = "./dummy"
    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
