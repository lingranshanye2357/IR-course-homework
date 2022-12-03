import pandas as pd
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List
from torch.nn.utils.rnn import pad_sequence


# 参数设置
base = "./Data/imdb"   # 这里换成数据的路径
pretrain_model = "bert-base-chinese"
max_length = 512
epochs = 3
seed = 900

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)


# 读取数据并进行预处理  注：这里需要知道中文训练的时候数据是什么形式，这里只是简单的复制粘贴了一下，要根据之后实际的数据形式进行修改
train = pd.read_csv(os.path.join(base, "labeledTrainData.tsv"), header=0, delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(base, "testData.tsv"), header=0, delimiter="\t", quoting=3)
train['review'] = train['review'].apply(lambda r: r.strip("\""))
test['review'] = test['review'].apply(lambda r: r.strip("\""))
examples = list(train['review'])+list(test['review'])


# 读取预训练模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrain_model)
model = BertForMaskedLM.from_pretrained(pretrain_model)


# 定义Dataset  要根据中文训练的实际情况修改
class LineByLineTextDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = tokenizer.batch_encode_plus(examples, add_special_tokens=True,
                                                    max_length=max_length, truncation=True)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


dataset = LineByLineTextDataset(examples, tokenizer, max_length)
print(" ".join(tokenizer.convert_ids_to_tokens(dataset[5])))


# 定义dataloader
def collate(examples: List[torch.Tensor]):
    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
    return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

dataloader = DataLoader(dataset, shuffle=True, batch_size=8, collate_fn=collate)


# 定义trainer
class Trainer:
    def __init__(self, model, dataloader, tokenizer, mlm_probability=0.15, lr=1e-4, with_cuda=True, cuda_devices=None,
                 log_freq=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.is_parallel = False
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.log_freq = log_freq

        # 多GPU训练  没有多GPU训练的条件，这里要根据实际情况修改
        if with_cuda and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUS for BERT")
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            self.is_parallel = True
        self.model.train()
        self.model.to(self.device)
        self.optim = AdamW(self.model.parameters(), lr=1e-4)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.dataloader)

    def iteration(self, epoch, dataloader, train=True):
        str_code = 'Train'
        total_loss = 0.0
        for i, batch in tqdm(enumerate(dataloader), desc="Training"):
            inputs, labels = self._mask_tokens(batch)
            inputs.to(self.device)
            labels.to(self.device)
            lm_loss, output = self.model(inputs, masked_lm_labels=labels)
            loss = lm_loss.mean()

            if train:
                self.model.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()

            total_loss += loss.item()
            post_fix = {
                "iter": i,
                "ave_loss": total_loss / (i + 1)
            }
            if i % self.log_freq == 0:
                print(post_fix)

        print(f"EP{epoch}_{str_code},avg_loss={total_loss / len(dataloader)}")

    def _mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Masked Language Model """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # 使用mlm_probability填充张量
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # 获取special token掩码
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        # 将special token位置的概率填充为0
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            # padding掩码
            padding_mask = labels.eq(tokenizer.pad_token_id)
            # 将padding位置的概率填充为0
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        # 对token进行mask采样
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # loss只计算masked

        # 80%的概率将masked token替换为[MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10%的概率将masked token替换为随机单词
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 余下的10%不做改变
        return inputs, labels


trainer = Trainer(model, dataloader, tokenizer)


# 训练和模型保存
for epoch in range(epochs):
    trainer.train(epoch)


model.save_pretrained(".")
