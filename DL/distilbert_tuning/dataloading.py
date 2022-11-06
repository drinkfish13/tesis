import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer

import pandas as pd

class DBertDataset(Dataset):
    def __init__(self, df, perc_keep_0):

        self.df = df
        self.df = self.df.reset_index().dropna(subset=["text", "label"])
        
        if perc_keep_0 < 1:
            df_1 = self.df[self.df["label"] == 1]
            df_0 = self.df[self.df["label"] == 0]
            df_0 = df_0.sample(int(perc_keep_0*len(df_0)), random_state=2)

            self.df = pd.concat([df_0, df_1])

    def __getitem__(self, i):

        text = self.df.iloc[i]["text"]
        label = self.df.iloc[i]["label"]
        return {"text": text, "label": label}

    def __len__(self):
        return len(self.df)

class DBertDataLoader:
    def __init__(self, df,
                perc_keep_0,
                 pretrained_name,
                 max_len=32,
                 drop_last=False,
                 batch_size=64,
                 shuffle=True
                 ):
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_name)
        self.max_len = max_len

        self.dataloader = DataLoader(
            dataset=DBertDataset(df, perc_keep_0),
            drop_last=drop_last,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate
        )

    def _collate(self, batch):
        tokenized_data = self.tokenizer.batch_encode_plus(
            [x["text"] for x in batch],
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )

        labels = torch.Tensor([x["label"] for x in batch]).long()
        return {
            "input_ids": tokenized_data["attention_mask"],
            "attention_mask": tokenized_data["attention_mask"],
            "labels": labels
        }



