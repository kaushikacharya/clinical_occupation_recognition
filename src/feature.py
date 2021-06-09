#!/usr/bin/env python3

"""
Feature extraction for a Document

References:
----------
    https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html
"""

import argparse
import logging
from src.document import *


class Feature:
    """
    Extraction of features of a Document class.
    """
    def __init__(self, doc_obj, logging_level=logging.INFO):
        """
        Parameters
        ----------
        doc_obj : Document class object
        logging_level : int
        """
        self.doc_obj = doc_obj
        # logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging_level)

    def extract_document_features(self):
        """
        Extract features for the document.

        document feature: List of sentence features
        sentence feature: List of word features stored in dict

        Returns
        -------
        list of list of dict
        """
        doc_features = []
        for sent_index in range(len(self.doc_obj.sentences)):
            sent_features = self.extract_sentence_features(sent_index=sent_index)
            doc_features.append(sent_features)

        self.logger.info("len(sentences): {}".format(len(self.doc_obj.sentences)))

        return doc_features

    def extract_sentence_features(self, sent_index):
        """Extract sentence features.

            Parameters:
            ----------
            sent_index : int

            Returns
            -------
            List of token features
        """
        sent_features = []

        start_token_index = self.doc_obj.sentences[sent_index].start_token_index
        end_token_index = self.doc_obj.sentences[sent_index].end_token_index
        self.logger.debug("sentence #{} :: count(tokens): {}".format(sent_index, end_token_index-start_token_index))

        for token_index in range(start_token_index, end_token_index):
            token_obj = self.doc_obj.tokens[token_index]

            token_features = dict()
            token_features["bias"] = 1.0
            token_text = self.doc_obj.text[token_obj.start_char_pos: token_obj.end_char_pos]
            token_features["token_lower"] = token_text.lower()
            token_features["is_caps"] = token_text.isupper()
            token_features["is_title"] = token_text.istitle()
            token_features["POS"] = self.doc_obj.tokens[token_index].part_of_speech
            token_features["dependency"] = self.doc_obj.tokens[token_index].dependency_tag

            if token_index == start_token_index:
                token_features["BOS"] = True
            else:
                token_features["prev_token_lower"] = sent_features[-1]["token_lower"]
                token_features["prev_token_POS"] = sent_features[-1]["POS"]

                sent_features[-1]["next_token_lower"] = token_features["token_lower"]
                sent_features[-1]["next_token_POS"] = token_features["POS"]

            head_token_index = token_obj.head_index
            if token_index == head_token_index:
                token_features["is_root_dependency_parse"] = True
            else:
                token_features["parent_dependency"] = self.doc_obj.tokens[head_token_index].dependency_tag
                token_features["parent_POS"] = self.doc_obj.tokens[head_token_index].part_of_speech

            # append token features to the list of sentence features
            sent_features.append(token_features)

        return sent_features


def main(args):
    import os

    assert args.logging_level in ["DEBUG", "INFO", "WARN", "WARNING", "ERROR",
                                  "CRITICAL"], "unexpected logging_level: {}".format(args.logging_level)
    logging_level = logging.getLevelName(level=args.logging_level)

    obj_nlp_process = NLPProcess(logging_level=logging_level)
    obj_nlp_process.load_nlp_model()
    doc_obj = Document(doc_case=args.clinical_case, logging_level=logging_level)
    doc_obj.read_document(document_file=os.path.join(args.data_dir, args.clinical_case + ".txt"))
    doc_obj.parse_document(nlp_process=obj_nlp_process)

    feature_obj = Feature(doc_obj=doc_obj, logging_level=args.logging_level)
    doc_features = feature_obj.extract_document_features()
    assert len(doc_features) == len(doc_obj.sentences), "Mismatch: count(sentences): {} :: len(doc_features): {]".format(len(doc_obj.sentences), len(doc_features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_case", action="store", dest="clinical_case")
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/MEDDOPROF/meddoprof-train-set/task1", dest="data_dir")
    parser.add_argument("--logging_level", action="store", default="INFO", dest="logging_level", help="options: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    args = parser.parse_args()

    main(args=args)
