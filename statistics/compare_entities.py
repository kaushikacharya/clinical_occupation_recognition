#!/usr/bin/env python

import argparse
import logging
import os

from src.annotation import *
from src.document import *
from src.nlp_process import NLPProcess
from src.utils import *


def main(args):
    assert args.logging_level in ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"], \
        "unexpected logging_level: {}".format(args.logging_level)
    logging_level = logging.getLevelName(level=args.logging_level)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    obj_nlp_process = NLPProcess(logging_level=logging_level)
    obj_nlp_process.load_nlp_model()

    if args.clinical_case:
        clinical_cases = [args.clinical_case]
    else:
        clinical_cases = collect_clinical_cases(data_dir=args.data_dir)

    for clinical_case in clinical_cases:
        if not os.path.exists(os.path.join(args.data_dir_ground_truth, clinical_case + ".ann")):
            logger.error("Missing ground truth annotation file for clinical case: {}".format(clinical_case))
            continue

        if not os.path.exists(os.path.join(args.data_dir_predict, clinical_case + ".ann")):
            logger.error("Missing predicted annotation file for clinical case: {}".format(clinical_case))
            continue

        logger.info("Processing clinical case: {}".format(clinical_case))
        doc_obj = Document(doc_case=clinical_case, logging_level=logging_level)
        doc_obj.read_document(document_file=os.path.join(args.data_dir, clinical_case + ".txt"))
        doc_obj.parse_document(nlp_process=obj_nlp_process)
        # read ground truth entity annotations
        entity_annotations_truth = read_annotation(
            ann_file=os.path.join(args.data_dir_ground_truth, clinical_case + ".ann"))
        doc_obj.logger.info("# entity annotations(ground truth): {}".format(len(entity_annotations_truth)))
        parse_annotations(entity_annotations=entity_annotations_truth, doc_obj=doc_obj)
        # read predicted entity annotations
        entity_annotations_predicted = read_annotation(
            ann_file=os.path.join(args.data_dir_predict, clinical_case + ".ann"))
        doc_obj.logger.info("# entity annotations(predicted): {}".format(len(entity_annotations_predicted)))
        parse_annotations(entity_annotations=entity_annotations_predicted, doc_obj=doc_obj)

        compare_annotations(entity_annotations_truth=entity_annotations_truth,
                            entity_annotations_predicted=entity_annotations_predicted, doc_obj=doc_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_case", action="store", default=None, dest="clinical_case")
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/MEDDOPROF/meddoprof_test_txt",
                        dest="data_dir", help="Directory containing report text")
    parser.add_argument("--data_dir_ground_truth", action="store", default="C:/KA/data/NLP/MEDDOPROF/meddoprof-test-GS/ner",
                        dest="data_dir_ground_truth", help="Directory containing ground truth annotations")
    parser.add_argument("--data_dir_predict", action="store", default=os.path.join(os.path.dirname(__file__), "../output/predict/meddoprof_test_txt"),
                        dest="data_dir_predict", help="Directory containing prediction annotations")
    parser.add_argument("--logging_level", action="store", default="INFO", dest="logging_level",
                        help="options: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    args = parser.parse_args()

    main(args=args)
