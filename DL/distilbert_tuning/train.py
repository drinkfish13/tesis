from runners import DBertRunner
from dataloading import DBertDataLoader
from models import DBertClassifier

import torch

import pandas as pd
import os
import json

import numpy as np

from collections import OrderedDict
from catalyst import dl

from utils import _fix_seeds

def train(args):
    _fix_seeds()

    ds_dir = args.data_dir
    train_df = pd.read_csv(os.path.join(ds_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(ds_dir, "test.csv"))

    with open(args.config_path, "r") as f:
        config = json.load(f)


    pretrained_name = config["pretrained_name"]
    max_len = config["max_len"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    layer_dropout = config["layer_dropout"]

    device = args.device
    n_classes = config["n_classes"]
    
    perc_keep_0 = config["perc_keep_0"]

    train_loader = DBertDataLoader(
        df=train_df, perc_keep_0=perc_keep_0,
        pretrained_name=pretrained_name,
        max_len=max_len,
        drop_last=False,
        batch_size=batch_size,
        shuffle=True
    ).dataloader
    test_loader = DBertDataLoader(
        df=test_df,perc_keep_0=1.0,
        pretrained_name=pretrained_name,
        max_len=max_len,
        drop_last=False,
        batch_size=batch_size,
        shuffle=False
    ).dataloader


    model = DBertClassifier(
        pretrained_name=pretrained_name,
        n_classes=n_classes,
        layer_dropout=layer_dropout
    )
    model = model.to(device)

    class_weights = None
    if config["weight"] == 'balanced':
        class_weights = len(train_df)/(n_classes * np.bincount(train_df["label"]))
        class_weights = torch.Tensor(class_weights)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # setting callbacks in order
    callbacks = OrderedDict({
        "criterion": dl.CriterionCallback(
            input_key="logits",
            target_key="labels",
            metric_key="loss"),

        "optimizer": dl.OptimizerCallback(
            metric_key="loss",
            accumulation_steps=1,
            grad_clip_params=None
        ),

        "accuracy": dl.AccuracyCallback(
            input_key="logits",
            target_key="labels",
            num_classes=n_classes
        ),
        "precision_recall_f1": dl.PrecisionRecallF1SupportCallback(
            input_key="logits",
            target_key="labels",
            num_classes=n_classes
        ),
        "checkpoint": dl.CheckpointCallback(
            logdir=args.logdir,
            loader_key="infer",
            metric_key="loss",
            minimize=True,
            save_n_best=0,
            metrics_filename="results.json"
        )
    })

   #  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam([
       {'params': model.bert_model.parameters(), 'lr': config['bert_lr']},
       {'params': model.classifier.parameters(), 'lr': config['cl_lr']}
       ], lr = config['def_lr'])

    runner = DBertRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders={
            "train": train_loader,
            "infer": test_loader
        },
        criterion=criterion,
        logdir=args.logdir,
        num_epochs=epochs,
        verbose=True,
        load_best_on_end=True,
        callbacks=callbacks,
        engine=dl.DeviceEngine(device)
    )

if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("-data_dir", type=str, default="/home/oleg/HateComplexity/prepared_data/hasoc_english/")
    args.add_argument("-logdir", type=str, default="/home/oleg/HateComplexity/Dbert/hasoc_english_new/")
    args.add_argument("-device", type=str, default="cuda")
    args.add_argument("-config_path", type=str, default="config.json")

    args = args.parse_args()

    train(args)







