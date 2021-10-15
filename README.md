# 用信息最大化的分层条件VAE从上下文中生成多样化和一致的QA对
这是**Pytorch的实现，**论文用信息最大化的分层条件VAE从语境中生成多样化和一致的QA对（**ACL 2020**，**长篇论文）。
[[Paper]](https://www.aclweb.org/anthology/2020.acl-main.20/) 
[[Slide]](https://drive.google.com/file/d/17oakiVKIaQ1Y_hSCkGfUIp8P6ORSYjjz/view?usp=sharing) 
[[Video]](https://slideslive.com/38928851/generating-diverse-and-consistent-qa-pairs-from-contexts-with-informationmaximizing-hierarchical-conditional-vaes).


## 简介
<img align="middle" width="800" src="https://github.com/seanie12/Info-HCVAE/blob/master/images/concept.png">
问答（QA）中最关键的挑战之一是标注数据的稀缺性，因为为目标文本领域获得带有人工标注的问题-答案（QA）对的成本很高。
解决这个问题的另一种方法是使用从问题背景或大量非结构化文本（如维基百科）中自动生成的QA对。
在这项工作中，我们提出了一个分层条件变分自编码（HCVAE），用于生成以非结构化文本为背景的QA对，同时最大化生成的QA对之间的互信息以确保其一致性。
我们在几个基准数据集上验证了我们的信息最大化分层条件变分自编码（InfoHCVAE），通过只使用生成的QA对（基于QA的评估）或使用生成的和人类标注的QA对（半监督学习）进行训练，
来评估QA模型（BERT-base）的性能，与先进的基线模型进行对比。
结果表明，我们的模型在这两项任务上都比所有基线模型获得了令人印象深刻的性能提升，只使用了一小部分数据进行训练。

这项工作的贡献__
- 我们提出了一个新颖的分层变分框架，用于从单一语境中生成不同的QA对，据我们所知，这是第一个用于生成问答对的概率生成模型（QAG）。
- 我们提出了一个InfoMax正则化项，通过最大化它们的互信息，有效地执行生成的QA对之间的一致性。这是一种解决QAG的QA对之间一致性的新方法。
- 我们在几个基准数据集上评估了我们的框架，方法是完全使用生成的QA对训练一个新的模型（基于QA的评估），或者同时使用ground truth和生成的QA对（半监督的QA）。我们的模型在这两项任务中都取得了令人印象深刻的表现，在很大程度上超过了现有的QAG基线。

## 依赖
This code is written in Python. Dependencies include
* python >= 3.6
* pytorch >= 1.4
* json-lines
* tqdm
* [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)
  如果安装了torch 1.8，那么conda install pytorch-scatter -c pyg 或者 pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.0+cu111.html
  如果安装了torch 1.9，那么安装方法: pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cpu.html
* [transfomers](https://github.com/huggingface/transformers)


## 下载 SQuAD 数据集
下载位置 [here](https://drive.google.com/file/d/1CdhslOycNFDwnDo7e8c7GaxvYxHrUlFG/view?usp=sharing). 
它包含SQuAD训练文件（data/squad/train-v1.1.json）和我们自己的dev/test分割文件（data/squad/my_dev.json, data/squad/my_test.json）。
我们对其进行预处理并转换为examples.pkl和features.pkl。这些pickle文件在data/pickle-file文件夹中。如果你想下载原始数据，请运行以下命令

```bash
mkdir squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O ./squad/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ./squad/dev-v1.1.json
```

## 训练 Info-HCVAE模型
用以下命令训练Info-HCVAE。checkpoint将被保存在 ./save/vae-checkpoint.
```bash
cd vae
python main.py --debug
```
## 生成问答对 
从未标注的段落生成QA对。如果你从SQuAD生成QA对，请使用选项-squad。
```bash
cd vae
python translate.py --data_file "DATA DIRECTORY for paragraph" --checkpoint "directory for Info-HCVAE model" --output_file "output file directory" --k "the number of QA pairs to sample for each paragraph" --ratio "the percentage of context to use"
```

## 问答对评估 (QAE)
它需要**3个1080ti GPUS（11GB内存）**来再现结果。你应该从[这里](https://drive.google.com/file/d/1CdhslOycNFDwnDo7e8c7GaxvYxHrUlFG/view?usp=sharing)下载数据并把它放在根目录下。
解压后，"data"文件夹包含QAE和半监督学习所需的所有文件。
```bash
cd qa-eval
python main.py --devices 0_1_2 --pretrain_file $PATH_TO_qaeval --unlabel_ratio 0.1 --lazy_loader --batch_size 24
```

## SQuAD的半监督学习
它需要**4个1080ti GPUS（11GB内存）**作为QAE，你应该从[这里]（https://drive.google.com/file/d/1CdhslOycNFDwnDo7e8c7GaxvYxHrUlFG/view?usp=sharing）下载数据，并将其放在根目录下。

```bash
cd qa-eval
python main.py --devices 0_1_2_3 --pretrain_file $PATH_TO_semieval --unlabel_ratio 1.0 --lazy_loader --batch_size 32
```

## 生成的问答对示例
从[这里](https://drive.google.com/file/d/1CdhslOycNFDwnDo7e8c7GaxvYxHrUlFG/view?usp=sharing)下载数据并解压到根目录下。
data/harv_synthetic_data_qae文件夹包含从Harvesting QA数据集中生成的QA对，没有经过任何过滤。
另一个文件夹data/harv_synthetic_data_semi包含相同的生成的QA对，但有后处理。如果F1低于阈值，我们用预训练的BERT QA模型替换生成的答案。


## 训练样本示例
```angular2html
{
  "context": "Architecturally, the school has a Catholic character. Atop the Main Building\u0027s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
  "qas": [
    {
      "answers": [
        {
          "answer_start": 515,
          "text": "Saint Bernadette Soubirous"
        }
      ],
      "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
      "id": "5733be284776f41900661182"
    },
    {
      "answers": [
        {
          "answer_start": 188,
          "text": "a copper statue of Christ"
        }
      ],
      "question": "What is in front of the Notre Dame Main Building?",
      "id": "5733be284776f4190066117f"
    },
    {
      "answers": [
        {
          "answer_start": 279,
          "text": "the Main Building"
        }
      ],
      "question": "The Basilica of the Sacred heart at Notre Dame is beside to which structure?",
      "id": "5733be284776f41900661180"
    },
    {
      "answers": [
        {
          "answer_start": 381,
          "text": "a Marian place of prayer and reflection"
        }
      ],
      "question": "What is the Grotto at Notre Dame?",
      "id": "5733be284776f41900661181"
    },
    {
      "answers": [
        {
          "answer_start": 92,
          "text": "a golden statue of the Virgin Mary"
        }
      ],
      "question": "What sits on top of the Main Building at Notre Dame?",
      "id": "5733be284776f4190066117e"
    }
  ]
}
```


## Reference
To cite the code/data/paper, please use this BibTex
```bibtex
@inproceedings{lee2020generating,
  title={Generating Diverse and Consistent QA pairs from Contexts with Information-Maximizing Hierarchical Conditional VAEs},
  author={Lee, Dong Bok and Lee, Seanie and Jeong, Woo Tae and Kim, Donghwan and Hwang, Sung Ju},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```
