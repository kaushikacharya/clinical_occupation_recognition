#!/usr/bin/env python

"""
Script to find out which ground truth entities are present in ProfNER gazetteer.
"""

import argparse
import glob
import io
import logging
import os

from src.annotation import *
from src.document import Document
from src.knowledge_base import Gazetteer
from src.nlp_process import NLPProcess


class GazetteerStatistics:
    def __init__(self, gazetteer_file, logging_level=logging.INFO):
        # Gazetteer trie
        self.gazetteer_obj = Gazetteer()
        self.gazetteer_obj.build_gazetteer_phrase_trie(filename=gazetteer_file)

        # logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging_level)

    def process_collection(self, data_dir, nlp_process):
        for f in glob.iglob(pathname=os.path.join(data_dir, "*.txt")):
            # extract clinical case
            file_basename, _ = os.path.splitext(os.path.basename(f))
            clinical_case = file_basename
            self.logger.info("Processing clinical case: {}".format(clinical_case))

            file_document = f
            file_ann = os.path.join(data_dir, clinical_case + ".ann")

            if not os.path.exists(file_ann):
                self.logger.info("Annotation file not available for clinical case: {}".format(clinical_case))
                continue

            try:
                doc_obj = Document(doc_case=clinical_case)
                doc_obj.read_document(document_file=file_document)
                doc_obj.parse_document(nlp_process=nlp_process)
                doc_entity_annotations = read_annotation(ann_file=file_ann)
                parse_annotations(entity_annotations=doc_entity_annotations, doc_obj=doc_obj)

                self.check_entity_in_gazetteer(doc_obj=doc_obj, entity_annotations=doc_entity_annotations)
            except Exception as err:
                self.logger.error("Failed for clinical case: {}".format(clinical_case), exc_info=True)

    def check_entity_in_gazetteer(self, doc_obj, entity_annotations):
        for ann_i in range(len(entity_annotations)):
            entity_type = entity_annotations[ann_i].type
            start_token_index = entity_annotations[ann_i].start_token_index
            end_token_index = entity_annotations[ann_i].end_token_index

            entity_tokens_text = []
            for token_index in range(start_token_index, end_token_index):
                start_char_pos_token = doc_obj.tokens[token_index].start_char_pos
                end_char_pos_token = doc_obj.tokens[token_index].end_char_pos
                entity_tokens_text.append(doc_obj.text[start_char_pos_token: end_char_pos_token])

            flag_entity_in_gazetteer = self.gazetteer_obj.search(entity_tokens_text)
            if flag_entity_in_gazetteer:
                self.logger.info("Entity Type: {} :: Found :: {}".format(entity_type, " ".join(entity_tokens_text)))
            else:
                sentence_index = entity_annotations[ann_i].sentence_index
                start_char_pos_sent = doc_obj.sentences[sentence_index].start_char_pos
                end_char_pos_sent = doc_obj.sentences[sentence_index].end_char_pos
                self.logger.info("Entity Type: {} :: Not Found :: {} :: Sentence: {}".format(
                    entity_type, " ".join(entity_tokens_text), doc_obj.text[start_char_pos_sent: end_char_pos_sent]))


def main(args):
    assert args.logging_level in ["DEBUG", "INFO", "WARN", "WARNING", "ERROR",
                                  "CRITICAL"], "unexpected logging_level: {}".format(args.logging_level)
    logging_level = logging.getLevelName(level=args.logging_level)

    obj_nlp_process = NLPProcess(logging_level=logging_level)
    obj_nlp_process.load_nlp_model()

    gazetteer_statistics = GazetteerStatistics(gazetteer_file=args.gazetteer_file, logging_level=args.logging_level)
    gazetteer_statistics.process_collection(data_dir=args.data_dir, nlp_process=obj_nlp_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/MEDDOPROF/meddoprof-train-set/task1/",
                        dest="data_dir")
    parser.add_argument("--logging_level", action="store", default="INFO", dest="logging_level",
                        help="options: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--gazetteer_file", action="store", default="C:/KA/data/NLP/MEDDOPROF/occupations-gazetteer/profner-gazetteer.tsv", dest="gazetteer_file")
    args = parser.parse_args()

    main(args=args)
