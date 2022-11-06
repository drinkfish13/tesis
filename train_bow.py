import pandas as pd


from bow_algos import BOWTrainer
import os
import json

from utils import _fix_seeds




def train_save(args):
    _fix_seeds()

    ds_dir = args.ds_dir
    ds_save_dir = args.save_dir

    # if args.resume:
    #     processed_datasets = os.listdir(save_dir)
    # else:
    #     processed_datasets = []

    # for p in os.listdir(dataset_dirs):
    #     ds_dir = os.path.join(dataset_dirs, p)
    #     if p not in processed_datasets:
    #         print("Starting: ", p)

    # ds_save_dir = os.path.join(save_dir, ds_dir)
    os.makedirs(ds_save_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(ds_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(ds_dir, "test.csv"))

    with open(os.path.join(ds_dir, 'add_config.json'), "r") as f:
        add_config = json.load(f)
        stop_words_lang = add_config["stop_words_lang"]
        use_bpe = add_config["use_bpe"]

    trainer = BOWTrainer(
        df=train_df,
        save_dir=ds_save_dir,
        stop_words_lang=stop_words_lang,
        use_bpe=use_bpe
    )

    trainer.fit_test(
        n_jobs=args.n_jobs,
        test_df=test_df,
        resume=args.resume
    )

    merged_test_results = [
        {
            "algo": k,
            "accuracy": v['accuracy'],
            "f1_score": v["f1_score"],
            "recall_score": v["recall_score"],
            "precision_score": v["precision_score"]
        } for k,v in trainer.test_results.items()
    ]
    merged_test_results = pd.DataFrame(merged_test_results)
    merged_test_results.to_csv(os.path.join(ds_save_dir, "merged_test_results.csv"))



if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("-ds_dir", type=str, default="prepared_data/")
    args.add_argument("-save_dir", type=str, default="output_bow/")
    args.add_argument("-n_jobs", type=int, default=8)
    args.add_argument("-resume", type=bool, default=True)

    args = args.parse_args()


    train_save(args)
