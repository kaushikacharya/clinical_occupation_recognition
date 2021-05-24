#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import traceback

from src.document import *
from src.feature import Feature
from src.nlp_process import NLPProcess

class CRF:
    def __init__(self, logging_level=logging.INFO):
        # logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging_level)

    def process_collection(self, data_dir, nlp_process_obj):
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

                X.extend(doc_features)

                # Populate document NER labels
                doc_named_entity_labels = []
                for sent in doc_obj.sentences:
                    sent_named_entity_labels = [doc_obj.tokens[token_index].ner_tag for token_index in
                                                range(sent.start_token_index, sent.end_token_index)]
                    doc_named_entity_labels.append(sent_named_entity_labels)

                y.extend(doc_named_entity_labels)

            except Exception as err:
                self.logger.error("Failed for clinical case: {}".format(clinical_case), exc_info=True)

        return X, y

def main(args):
    assert args.logging_level in ["DEBUG", "INFO", "WARN", "WARNING", "ERROR",
                                  "CRITICAL"], "unexpected logging_level: {}".format(args.logging_level)
    logging_level = logging.getLevelName(level=args.logging_level)

    obj_nlp_process = NLPProcess(logging_level=logging_level)
    obj_nlp_process.load_nlp_model()
    obj_crf = CRF(logging_level=logging_level)
    X_train, y_train = obj_crf.process_collection(data_dir=args.data_dir, nlp_process_obj=obj_nlp_process)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/MEDDOPROF/meddoprof-train-set/task1", dest="data_dir")
    parser.add_argument("--logging_level", action="store", default="INFO", dest="logging_level", help="options: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    args = parser.parse_args()

    main(args=args)
