#!/usr/bin/env python3

import argparse
import glob
import joblib
import logging
import numpy as np
import os
from pathlib import Path
import seqeval.metrics as seqeval_metrics
import shutil
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

    def process_collection(self, data_dir, nlp_process):

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
                doc_obj = Document(doc_case=clinical_case)
                doc_obj.read_document(document_file=file_document)
                doc_obj.parse_document(nlp_process=nlp_process)
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

    def predict_collection(self, input_data_dir, output_data_dir, nlp_process):
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)

        for f in glob.iglob(pathname=os.path.join(input_data_dir, "*.txt")):
            # extract clinical case
            file_basename, _ = os.path.splitext(os.path.basename(f))
            clinical_case = file_basename

            try:
                doc_obj = Document(doc_case=clinical_case)
                doc_obj.read_document(document_file=f)
                doc_obj.parse_document(nlp_process=nlp_process)

                y_pred_doc = self.predict_document(doc_obj=doc_obj)
                entity_annotations = self.build_predicted_entities(doc_obj=doc_obj, y_pred_doc=y_pred_doc)
                file_ann = os.path.join(output_data_dir, clinical_case+".ann")
                self.save_predicted_entities(entity_annotations=entity_annotations, doc_obj=doc_obj, filename=file_ann)
            except Exception as err:
                self.logger.error("Failed for clinical case: {}".format(clinical_case), exc_info=True)

    def predict_document(self, doc_obj):
        """Predict NER tags for the document"""
        feature_obj = Feature(doc_obj=doc_obj)
        doc_features = feature_obj.extract_document_features()
        y_pred_doc = self.crf.predict(X=doc_features)

        return y_pred_doc

    @staticmethod
    def build_predicted_entities(doc_obj, y_pred_doc):
        entity_annotations = []

        assert len(y_pred_doc) == len(doc_obj.sentences),\
            "Mismatch: len(predicted sentences): {} :: len(document sentences): {}".format(len(y_pred_doc), len(doc_obj.sentences))

        # iterate over each of the sentences and build annotations(if entity predicted)
        for sent_i in range(len(doc_obj.sentences)):
            start_token_index_sent = doc_obj.sentences[sent_i].start_token_index
            end_token_index_sent = doc_obj.sentences[sent_i].end_token_index
            y_pred_sent = y_pred_doc[sent_i]
            assert len(y_pred_sent) == (end_token_index_sent - start_token_index_sent),\
                "Mismatch: sent_i: {} :: len(predicted tokens): {} :: len(sentence tokens): {}".format(sent_i, len(y_pred_sent), (end_token_index_sent - start_token_index_sent))

            # iterate over the tokens
            # N.B. start_token_index, end_token_index for the document sentences are w.r.t. token indexes of the entire documnent
            for token_i in range(end_token_index_sent - start_token_index_sent):
                ner_tag = y_pred_sent[token_i]
                if ner_tag == "O":
                    continue

                ner_tag_tokens = ner_tag.split("-")
                assert len(ner_tag_tokens) > 1, "NER tag not in BIO format :: Named Entity: {}".format(ner_tag)

                if ner_tag_tokens[0] == "B":
                    # start of next named entity
                    entity_type = ner_tag[2:]
                    entity_id = "T" + str(len(entity_annotations) + 1)
                    # token index wrt tokens of the entire document
                    start_token_index_ent = start_token_index_sent + token_i
                    # Currently assigning token end as the next token. But will be modified if entity continues to next token(s)  # noqa
                    end_token_index_ent = start_token_index_ent + 1
                    start_char_pos_ent = doc_obj.tokens[start_token_index_ent].start_char_pos
                    end_char_pos_ent = doc_obj.tokens[start_token_index_ent].end_char_pos

                    entity_ann = EntityAnnotation(entity_id=entity_id, start_char_pos=start_char_pos_ent,
                                                  end_char_pos=end_char_pos_ent, start_token_index=start_token_index_ent,
                                                  end_token_index=end_token_index_ent, entity_type=entity_type)
                    entity_annotations.append(entity_ann)
                elif ner_tag_tokens[0] == "I":
                    # Continuation of previous named entity
                    # Update end token index and end char pos. Would be updated again if next token is also part of the current entity. # noqa
                    end_token_index_ent = start_token_index_sent + token_i + 1
                    end_char_pos_ent = doc_obj.tokens[end_token_index_ent-1].end_char_pos
                    entity_annotations[-1].end_token_index = end_token_index_ent
                    entity_annotations[-1].end_char_pos = end_char_pos_ent
                else:
                    assert False, "ERROR :: ner_tag expected either B-<Entity Type> or I-<Entity Type>"

        return entity_annotations

    @staticmethod
    def save_predicted_entities(entity_annotations, doc_obj, filename):
        with io.open(file=filename, mode="w", encoding="utf-8") as fd:
            for entity_ann in entity_annotations:
                entity_text = doc_obj.text[entity_ann.start_char_pos: entity_ann.end_char_pos]
                fd.write("{}\t{} {} {}\t{}\n".format(entity_ann.id, entity_ann.type, entity_ann.start_char_pos,
                                                     entity_ann.end_char_pos, entity_text))


def main(args):
    assert args.logging_level in ["DEBUG", "INFO", "WARN", "WARNING", "ERROR",
                                  "CRITICAL"], "unexpected logging_level: {}".format(args.logging_level)
    logging_level = logging.getLevelName(level=args.logging_level)

    obj_nlp_process = NLPProcess(logging_level=logging_level)
    obj_nlp_process.load_nlp_model()
    obj_crf = CRF(logging_level=logging_level)

    X = None
    y = None

    if args.flag_train or args.flag_train_test_split or args.flag_evaluate:
        start_time = time.time()
        X, y = obj_crf.process_collection(data_dir=args.data_dir, nlp_process=obj_nlp_process)
        obj_crf.logger.info("\nprocess_collection() on data_dir took {:.3f} seconds\n".format(time.time() - start_time))

    if args.flag_train:
        # Select entire or subset of train data
        if args.train_size is not None and args.train_size < 1:
            train_index_arr, _ = train_test_split(range(len(y)), train_size=args.train_size, random_state=args.random_seed)
            train_X = [X[i] for i in train_index_arr]
            train_y = [y[i] for i in train_index_arr]
        else:
            train_X = X
            train_y = y

        start_time = time.time()
        obj_crf.train(X_train=train_X, y_train=train_y)
        obj_crf.logger.info("\ntrain CRF took {:.3f} seconds :: train_size: {}\n".format(time.time() - start_time, args.train_size))

        # Dump the trained model
        obj_crf.save_model(filename=args.train_model)
        obj_crf.logger.info("Train model: {} saved".format(args.train_model))

    if args.flag_train_test_split:
        assert args.train_size is not None, "Expected float value in the range [0,1] for train_size"
        assert 0 <= args.train_size <= 1, "train_size: {} expected value in [0,1]".format(args.train_size)
        train_size = args.train_size
        while train_size < 1.0:
            # split data into train and test set
            # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
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

    if args.flag_predict:
        obj_crf.load_model(filename=args.train_model)
        obj_crf.logger.info("Train model: {} loaded".format(args.train_model))

        start_time = time.time()
        output_dir = os.path.join(os.path.dirname(__file__), "../output/predict", os.path.basename(Path(args.data_dir)))
        # Delete output dir (if exists)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        obj_crf.predict_collection(input_data_dir=args.data_dir, output_data_dir=output_dir, nlp_process=obj_nlp_process)
        print('\nPredict took {:.3f} seconds\n'.format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/MEDDOPROF/meddoprof-train-set/task1/", dest="data_dir")
    parser.add_argument("--logging_level", action="store", default="INFO", dest="logging_level",
                        help="options: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--train_size", action="store", type=float, default=None, dest="train_size",
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
    parser.add_argument("--flag_predict", action="store_true", default=False, dest="flag_predict")

    args = parser.parse_args()

    main(args=args)
