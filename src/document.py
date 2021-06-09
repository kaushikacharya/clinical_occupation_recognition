#!/usr/bin/env python

import argparse
import io
import logging

from src.annotation import EntityAnnotation
from src.nlp_process import NLPProcess


class Token:
    """Token class.
    """
    def __init__(self,
                 start_char_pos=None,
                 end_char_pos=None,
                 lemma=None,
                 part_of_speech=None,
                 dependency_tag=None,
                 head_index=None,
                 children_index_arr=None,
                 ner_tag=None):
        # char position offsets in the document text
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos
        self.lemma = lemma
        self.part_of_speech = part_of_speech
        self.dependency_tag = dependency_tag
        self.head_index = head_index
        self.children_index_arr = children_index_arr
        self.ner_tag = ner_tag


class Sentence:
    """Sentence class.

        References
        ----------
        https://spacy.io/usage/linguistic-features#sbd
        - Sentence segmentation
    """
    def __init__(self, start_char_pos=None, end_char_pos=None, start_token_index=None, end_token_index=None):
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos
        self.start_token_index = start_token_index
        self.end_token_index = end_token_index


class Document:
    """Clinical case document"""
    def __init__(self, doc_case, logging_level=logging.INFO):
        self.case = doc_case
        self.text = None
        # objects of Sentence class
        self.sentences = []
        # objects of Token class
        self.tokens = []
        # logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging_level)
        # print(self.logger.isEnabledFor(level=logging_level))

    def read_document(self, document_file):
        try:
            with io.open(document_file, mode="r", encoding="utf-8") as fd:
                text = fd.read()
            self.text = text
        except Exception as err:
            print(err)

    def parse_document(self, nlp_process):
        """
        Populate sentence and tokens for the document.
        NLPProcess is also applied to extract part-of-speech and dependency parse.
        """
        assert self.text is not None, "pre-requisite: execute read_document()"
        doc = nlp_process.construct_doc(text=self.text)

        for sent_i, sent in enumerate(doc.sents):
            # start_token_index_sent = len(self.tokens)
            for token in sent:
                end_char_pos_token = token.idx+len(token.text)
                token_obj = Token(start_char_pos=token.idx, end_char_pos=end_char_pos_token, lemma=token.lemma_,
                                  part_of_speech=token.pos_, dependency_tag=token.dep_, head_index=token.head.i,
                                  children_index_arr=[child.i for child in token.children])
                self.tokens.append(token_obj)

            # end_token_index_sent = len(self.tokens)
            start_char_pos_sent = doc[sent.start].idx
            end_char_pos_sent = start_char_pos_sent + len(sent.text)
            sentence_obj = Sentence(start_char_pos=start_char_pos_sent, end_char_pos=end_char_pos_sent,
                                    start_token_index=sent.start, end_token_index=sent.end)
            self.sentences.append(sentence_obj)
            self.logger.debug("Sentence #{} :: char pos range: ({},{}) :: token range: ({},{}) :: text: {}".format(
                sent_i, start_char_pos_sent, end_char_pos_sent, sent.start, sent.end, sent.text))

        self.logger.info("# sentences: {}".format(len(self.sentences)))
        self.logger.info("# tokens: {}".format(len(self.tokens)))

    def assign_ground_truth_ner_tags(self, entity_annotations):
        """Assign NER ground truth tag to the tokens"""

        # First assign tag "O" to all the tokens
        ner_tag = "O"
        for token_i in range(len(self.tokens)):
            self.tokens[token_i].ner_tag = ner_tag

        # Now update tag for the tokens which are part of entity annotations
        for ann_i in range(len(entity_annotations)):
            start_token_index = entity_annotations[ann_i].start_token_index
            end_token_index = entity_annotations[ann_i].end_token_index
            self.tokens[start_token_index].ner_tag = "B-" + entity_annotations[ann_i].type
            for token_index in range(start_token_index+1, end_token_index):
                self.tokens[token_index].ner_tag = "I-" + entity_annotations[ann_i].type


def main(args):
    import os
    from src.annotation import read_annotation, parse_annotations

    assert args.logging_level in ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"],\
        "unexpected logging_level: {}".format(args.logging_level)
    logging_level = logging.getLevelName(level=args.logging_level)

    obj_nlp_process = NLPProcess(logging_level=logging_level)
    obj_nlp_process.load_nlp_model()
    doc_obj = Document(doc_case=args.clinical_case, logging_level=logging_level)
    doc_obj.read_document(document_file=os.path.join(args.data_dir, args.clinical_case+".txt"))
    doc_obj.parse_document(nlp_process=obj_nlp_process)
    entity_annotations = read_annotation(ann_file=os.path.join(args.data_dir, args.clinical_case+".ann"))
    doc_obj.logger.info("# entity annotations: {}".format(len(entity_annotations)))
    parse_annotations(entity_annotations=entity_annotations, doc_obj=doc_obj)
    doc_obj.assign_ground_truth_ner_tags(entity_annotations=entity_annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_case", action="store", dest="clinical_case")
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/MEDDOPROF/meddoprof-train-set/task1", dest="data_dir")
    parser.add_argument("--logging_level", action="store", default="INFO", dest="logging_level", help="options: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    args = parser.parse_args()

    main(args=args)
