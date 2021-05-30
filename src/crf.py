#!/usr/bin/env python3

import argparse
import glob
import joblib
import logging
import numpy as np
import os
import seqeval.metrics as seqeval_metrics
import sklearn_crfsuite
import sklearn_crfsuite.metrics as crfsuite_metrics
from sklearn.model_selection import train_test_split
import time
import traceback

from src.document import *
from src.feature import Feature
from src.nlp_process import NLPProcess


class CRF:
    def __init__(self, logging_level=logging.INFO):
        self.crf = None
        # logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging_level)

    def process_collection(self, data_dir, nlp_process_obj, random_seed=0):

        # set seed for the pseudo-random number generator
        np.random.seed(random_seed)

        X = []  # features
        y = []  # NER labels

        for f in glob.iglob(pathname=os.path.join(data_dir, "*.txt")):
            # extract clinical case
            file_basename, _ = os.path.splitext(os.path.basename(f))
            clinical_case = file_basename
            self.logger.info("Processing clinical case: {}".format(clinical_case))

            file_document = f
            file_ann = os.path.join(data_dir, clinical_case+".ann")

            if not os.path.exists(file_ann):
                self.logger.info("Annotation file not available for clinical case: {}".format(clinical_case))
                continue

            try:
                doc_obj = Document(doc_case=clinical_case, nlp_process_obj=nlp_process_obj)
                doc_obj.read_document(document_file=file_document)
                doc_obj.parse_document()
                doc_obj.read_annotation(ann_file=file_ann)
                doc_obj.parse_annotations()
                doc_obj.assign_ground_truth_ner_tags()

                feature_obj = Feature(doc_obj=doc_obj)
                doc_features = feature_obj.extract_document_features()

                # Populate document NER labels
                doc_named_entity_labels = []
                for sent in doc_obj.sentences:
                    sent_named_entity_labels = [doc_obj.tokens[token_index].ner_tag for token_index in
                                                range(sent.start_token_index, sent.end_token_index)]
                    doc_named_entity_labels.append(sent_named_entity_labels)

                X.extend(doc_features)
                y.extend(doc_named_entity_labels)

            except Exception as err:
                self.logger.error("Failed for clinical case: {}".format(clinical_case), exc_info=True)

        return X, y

    def train(self, X_train, y_train, algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True):
        self.logger.info("Train begin")
        self.crf = sklearn_crfsuite.CRF(algorithm=algorithm, c1=c1, c2=c2, max_iterations=max_iterations,
                                        all_possible_transitions=all_possible_transitions,
                                        verbose=self.logger.level == logging.DEBUG)
        self.crf.fit(X=X_train, y=y_train)
        self.logger.info("Train end")

    def save_model(self, filename):
        """Save trained model on disk"""
        assert self.crf is not None, "pre-requisite: Execute train()"
        model_dir = os.path.dirname(filename)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        joblib.dump(value=self.crf, filename=filename)
        self.logger.info("Model saved: {}".format(filename))

    def load_model(self, filename):
        """Load trained model"""
        self.crf = joblib.load(filename=filename)

    def evaluate(self, X_test, y_test):
        """Evaluate named entity predictions against gold standard.
                Compared at both token as well as entity level.
        """
        assert self.crf is not None, "Pre-requisite: Either execute train() or load_model()"
        y_pred = self.crf.predict(X=X_test)

        labels = list(self.crf.classes_)
        labels.remove('O')
        print("labels: {}".format(labels))

        print("\n---- Token level metrics ----\n")
        f1_score = crfsuite_metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
        print("F1: {}".format(f1_score))

        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        sorted_labels.append('O')

        print(crfsuite_metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        ))

        print("\n---- Entity level metrics ----\n")
        print("Precision: {}".format(seqeval_metrics.precision_score(y_true=y_test, y_pred=y_pred)))
        print("Recall: {}".format(seqeval_metrics.recall_score(y_true=y_test, y_pred=y_pred)))
        print("F1: {}".format(seqeval_metrics.f1_score(y_true=y_test, y_pred=y_pred)))

        print(seqeval_metrics.classification_report(y_true=y_test, y_pred=y_pred, digits=3))


def main(args):
    assert args.logging_level in ["DEBUG", "INFO", "WARN", "WARNING", "ERROR",
                                  "CRITICAL"], "unexpected logging_level: {}".format(args.logging_level)
    logging_level = logging.getLevelName(level=args.logging_level)
    assert 0 <= args.train_size <= 1, "train_size: {} expected value in [0,1]".format(args.train_size)

    obj_nlp_process = NLPProcess(logging_level=logging_level)
    obj_nlp_process.load_nlp_model()
    obj_crf = CRF(logging_level=logging_level)

    start_time = time.time()
    X, y = obj_crf.process_collection(data_dir=args.data_dir, nlp_process_obj=obj_nlp_process, random_seed=args.random_seed)
    obj_crf.logger.info("\nprocess_collection() on data_dir took {:.3f} seconds\n".format(time.time() - start_time))
    start_time = time.time()

    if args.flag_train:
        obj_crf.train(X_train=X, y_train=y)
        obj_crf.logger.info("\ntrain CRF took {:.3f} seconds\n".format(time.time() - start_time))

        # Dump the trained model
        obj_crf.save_model(filename=args.train_model)
        obj_crf.logger.info("Train model: {} saved".format(args.train_model))

    if args.flag_train_test_split:
        train_size = args.train_size
        while train_size < 1.0:
            # split data into train and test set
            train_X, dev_X, train_y, dev_y = train_test_split(X, y, train_size=train_size, random_state=args.random_seed)
            obj_crf.logger.info("{}Train size(fraction): {} :: len(train samples): {} :: len(test samples): {} {}\n"
                                .format("-"*5, train_size, len(train_y), len(dev_y), "-"*5))

            # train on train set
            obj_crf.train(X_train=train_X, y_train=train_y)

            # evaluate on train set
            obj_crf.logger.info("Evaluate on train set")
            obj_crf.evaluate(X_test=train_X, y_test=train_y)

            # evaluate on test set
            obj_crf.logger.info("Evaluate on test set")
            obj_crf.evaluate(X_test=dev_X, y_test=dev_y)

            train_size += args.train_size

        # train on entire dataset
        obj_crf.train(X_train=X, y_train=y)

        # evaluate on entire dataset
        obj_crf.logger.info("Evaluate on entire set using the model trained on the same set")
        obj_crf.evaluate(X_test=X, y_test=y)

    if args.flag_evaluate:
        obj_crf.load_model(filename=args.train_model)
        obj_crf.logger.info("Train model: {} loaded".format(args.train_model))

        start_time = time.time()
        obj_crf.evaluate(X_test=X, y_test=y)
        print('\nEvaluate took {:.3f} seconds\n'.format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/MEDDOPROF/meddoprof-train-set/task1", dest="data_dir")
    parser.add_argument("--logging_level", action="store", default="INFO", dest="logging_level",
                        help="options: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--train_size", action="store", type=float, default=0.2, dest="train_size",
                        help="Fraction of data_dir for training the model."
                             " Values should range in [0,1] (both ends included)."
                             " Complementary fraction would be used for validation.")
    train_model = os.path.join(os.path.dirname(__file__), "../output/models/crf_model.pkl")
    parser.add_argument("--train_model", action="store", default=train_model,
                        dest="train_model", help="train model file path")
    parser.add_argument("--flag_train", action="store_true", default=False, dest="flag_train")
    parser.add_argument("--flag_train_test_split", action="store_true", default=False, dest="flag_train_test_split",
                        help="a) Split data (b) train model on train split  (c) evaluate on both train and test split")
    parser.add_argument("--random_seed", action="store", type=int, default=0, dest="random_seed")
    parser.add_argument("--flag_evaluate", action="store_true", default=False, dest="flag_evaluate",
                        help="Evaluate on the dataset using the train_model")

    args = parser.parse_args()

    main(args=args)
