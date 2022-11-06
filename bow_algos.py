from utils import _fix_seeds

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

import numpy as np
from scipy.sparse import issparse

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from testing import Testing

import os
import json
from tqdm import tqdm
from joblib import dump

from stop_words import get_stop_words
from sklearn.metrics import log_loss, make_scorer

class DenseTransformer():

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if issparse(X):
            return X.todense()
        return X


class BOWTrainer:

    def __init__(self, df, save_dir=None, stop_words_lang=None, use_bpe=True, label2name=None):

        """
        :param df: --pd.DataFrame with columns: text, label. Label should be already encoded.
        :param save_dir: --str, path to savedir
        :param stop_words_lang: --str, lang to load stop words for
        """

        _fix_seeds()

        self.df = df
        self.df = self.df[["text", "label"]].dropna()

        self.logreg_param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"], "random_state": [2]}

        self.forest_param_grid = {
            "class_weight": ["balanced", "balanced_subsample"],
            'max_depth': [10, 50, 100 ],
            'min_samples_leaf': [2, 4],
            'min_samples_split': [2, 5],
            'n_estimators': [100, 300, 400],
            "random_state": [2]
        }

        self.svm_param_grid = {'C':np.logspace(-3, 3, 7),'gamma':["auto", "scale"],
                               'kernel':['linear','rbf'], "random_state": [2]}
        self.nb_param_grid = {'var_smoothing': np.logspace(0,-9, num=10)}

        self.best_ppls = {}
        self.test_results = {}

        # fitting bpe
        if use_bpe:
            self.tokenizer = Tokenizer(BPE())
            self.trainer = BpeTrainer(special_tokens=[])
            self.tokenizer.train_from_iterator(df["text"].values.tolist(), trainer=self.trainer)
            tokenizer_params = [None, self.tokenizer.encode]
        else:
            tokenizer_params = [None]

        self.tfidf_grid = {
            "tokenizer": tokenizer_params,
            "stop_words": [None],
            "ngram_range": [(1,1), (1,2), (1,3), (1,4)],
            "max_df": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_df": [1, 0.1, 0.2, 0.3, 0.4],
            "binary": [True, False],
            "norm": ['l1', 'l2'],
            "use_idf": [True, False],
        }

        if stop_words_lang:

            self.tfidf_grid["stop_words"].append(
                get_stop_words(stop_words_lang)
            )

        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self.label2name = label2name

    def model_names(self):
        return list(self.best_ppls.keys())


    def fit_test(self, n_jobs=-1, verbose=2, test_df=(), resume=True):

        trained_algos = []
        if self.save_dir:
            trained_algos = os.listdir(self.save_dir)

        for algo_grid, algo in tqdm([
            (self.forest_param_grid, RandomForestClassifier()),
          #commented as needed only forest
            #  (self.logreg_param_grid, LogisticRegression()),
          #  (self.nb_param_grid, GaussianNB()),
          #  (self.svm_param_grid, SVC())
        ]):

            algo_name = type(algo).__name__

            if algo_name not in trained_algos and resume:

                pipeline = Pipeline(
                    steps=[('preprocessing', TfidfVectorizer()),
                           ('to_dense', DenseTransformer()),
                           ('algo', algo)]
                )

                params_grid = {}
                for k, vals in self.tfidf_grid.items():
                    params_grid[f"preprocessing__{k}"] = vals
                for k, vals in algo_grid.items():
                    params_grid[f"algo__{k}"] = vals

                search = GridSearchCV(estimator=pipeline,
                                      cv=3,
                                      param_grid=params_grid,
                                      n_jobs=n_jobs,
                                      scoring=make_scorer(log_loss, greater_is_better=False),
                                      verbose=verbose
                                      )
                search.fit(self.df["text"], self.df["label"])


                if self.save_dir:

                    algo_save_dir = os.path.join(self.save_dir, algo_name)
                    os.makedirs(algo_save_dir, exist_ok=True)
                    with open(os.path.join(algo_save_dir, "best_params.json"), "w") as f:
                        json.dump(search.best_params_, f)
                    with open(os.path.join(algo_save_dir, "best_score.json"), "w") as f:
                        json.dump({"score": search.best_score_}, f)

                    dump(search.best_estimator_, os.path.join(algo_save_dir, 'algo_pipeline.joblib'))
                    self.best_ppls[algo_name] = search.best_estimator_

                    if len(test_df):
                        preds = search.best_estimator_.predict(test_df["text"])
                        test_res = Testing.test(
                            pred_classes=preds,
                            labels=test_df["label"],
                            target_names=self.label2name
                        )
                        with open(os.path.join(algo_save_dir, "test_res.json"), "w") as f:
                            json.dump(test_res, f)
                        self.test_results[algo_name] = test_res

            if resume and algo_name in trained_algos:
                print(f"{algo_name} is already trained.\n")




    def predict(self, df, model_name=None):

        output = {}

        items = self.best_ppls.items()
        if model_name:
            if model_name not in self.best_ppls:
                raise Exception(f"{model_name} is not supported. Check supported names with BOWTrainer.model_names()")

            items = (model_name, self.best_ppls[model_name])


        for pipeline_name, ppl in items:

            classes = ppl.predict(df["text"])
            probs = ppl.predict_proba(df["text"])

            output[pipeline_name] = {
                "classes": classes,
                "probs": probs
            }
        return output
