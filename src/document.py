#!/usr/bin/env python

import argparse
import io
import logging
import re

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
    def __init__(self, doc_case, nlp_process_obj, logging_level=logging.INFO):
        self.case = doc_case
        self.text = None
        # objects of Sentence class
        self.sentences = []
        # objects of Token class
        self.tokens = []
        self.entity_annotations = []
        # logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging_level)
        # print(self.logger.isEnabledFor(level=logging_level))

        self.nlp_process = nlp_process_obj

    def read_document(self, document_file):
        try:
            with io.open(document_file, mode="r", encoding="utf-8") as fd:
                text = fd.read()
            self.text = text
        except Exception as err:
            print(err)

    def parse_document(self):
        """
        Populate sentence and tokens for the document.
        """
        assert self.text is not None, "pre-requisite: execute read_document()"
        doc = self.nlp_process.construct_doc(text=self.text)

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

    def read_annotation(self, ann_file):
        """
        Read brat standoff formatted annotation file.

        References
        ----------
        https://brat.nlplab.org/standoff.html
        """
        with io.open(file=ann_file, mode="r", encoding="utf-8") as fd:
            for line in fd:
                line = line.strip()

                if line == "" or line[0] != "T":
                    continue

                # split the line using tab delimiter
                tokens = line.split("\t")
                assert len(tokens) == 3, "Expected format: <Entity ID><TAB><Entity Type><Space><Begin char pos><Space><End char pos><TAB><Text> : {}".format(line)

                entity_id = tokens[0]
                # extract entity type and char position range
                char_index = tokens[1].find(" ")
                assert char_index > 0, "space not found in middle token :: line: {}".format(line)
                entity_type = tokens[1][:char_index]
                entity_char_pos_arr = [int(x) for x in re.findall(r'\d+', tokens[1][char_index:])]
                assert len(entity_char_pos_arr) > 1, "Expected at least two elements for char position range"
                if len(entity_char_pos_arr) > 2:
                    print("Discontinuous entity annotation: {}".format(line))

                entity_ann = EntityAnnotation(entity_id=entity_id, start_char_pos=entity_char_pos_arr[0],
                                              end_char_pos=entity_char_pos_arr[1], entity_type=entity_type)
                self.entity_annotations.append(entity_ann)

        # Sort wrt start_char_pos
        self.entity_annotations = sorted(self.entity_annotations, key=lambda x: x.start_char_pos)

        self.logger.info("# entity annotations: {}".format(len(self.entity_annotations)))

    def parse_annotations(self):
        """Extract and assign token range to the entity annotations."""
        entity_ann_index = 0
        sent_i = 0

        while (sent_i < len(self.sentences)) and (entity_ann_index < len(self.entity_annotations)):
            start_char_pos_sent = self.sentences[sent_i].start_char_pos
            end_char_pos_sent = self.sentences[sent_i].end_char_pos
            start_char_pos_ann = self.entity_annotations[entity_ann_index].start_char_pos
            end_char_pos_ann = self.entity_annotations[entity_ann_index].end_char_pos

            if end_char_pos_sent <= start_char_pos_ann:
                # move to the next sentence
                sent_i += 1
                continue
            elif start_char_pos_sent <= start_char_pos_ann < end_char_pos_sent:
                # Current entity belongs to the current sentence
                # Identify token range for the current entity
                start_token_index_ann = None
                end_token_index_ann = None

                token_index = self.sentences[sent_i].start_token_index
                while token_index < self.sentences[sent_i].end_token_index:
                    start_char_pos_token = self.tokens[token_index].start_char_pos
                    end_char_pos_token = self.tokens[token_index].end_char_pos

                    if start_char_pos_token <= start_char_pos_ann < end_char_pos_token:
                        start_token_index_ann = token_index
                        break

                    # increment token index
                    token_index += 1

                assert start_token_index_ann is not None, "start_token_index_ann not assigned. Sentence #{}".format(sent_i)
                while token_index < self.sentences[sent_i].end_token_index:
                    start_char_pos_token = self.tokens[token_index].start_char_pos
                    end_char_pos_token = self.tokens[token_index].end_char_pos

                    if end_char_pos_ann > end_char_pos_token:
                        # move to the next token
                        token_index += 1
                    else:
                        end_token_index_ann = token_index + 1
                        break

                if end_token_index_ann is None:
                    end_token_index_ann = self.sentences[sent_i].end_token_index

                assert self.entity_annotations[entity_ann_index].start_token_index is None, "start_token_index already assigned for entity #{}".format(entity_ann_index)
                assert self.entity_annotations[entity_ann_index].end_token_index is None, "end_token_index already assigned for entity #{}".format(entity_ann_index)

                # Assign the token index range to the entity annotation
                self.entity_annotations[entity_ann_index].start_token_index = start_token_index_ann
                self.entity_annotations[entity_ann_index].end_token_index = end_token_index_ann

                # move to the next entity
                entity_ann_index += 1
            else:
                assert False, "entity_ann_index: {} missed assignment of token range".format(entity_ann_index)

    def assign_ground_truth_ner_tags(self):
        """Assign NER ground truth tag to the tokens"""

        # First assign tag "O" to all the tokens
        ner_tag = "O"
        for token_i in range(len(self.tokens)):
            self.tokens[token_i].ner_tag = ner_tag

        for ann_i in range(len(self.entity_annotations)):
            start_token_index = self.entity_annotations[ann_i].start_token_index
            end_token_index = self.entity_annotations[ann_i].end_token_index
            self.tokens[start_token_index].ner_tag = "B-" + self.entity_annotations[ann_i].type
            for token_index in range(start_token_index+1, end_token_index):
                self.tokens[token_index].ner_tag = "I-" + self.entity_annotations[ann_i].type


def main(args):
    import os

    assert args.logging_level in ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"], "unexpected logging_level: {}".format(args.logging_level)
    logging_level = logging.getLevelName(level=args.logging_level)

    obj_nlp_process = NLPProcess(logging_level=logging_level)
    obj_nlp_process.load_nlp_model()
    doc_obj = Document(doc_case=args.clinical_case, nlp_process_obj=obj_nlp_process, logging_level=logging_level)
    doc_obj.read_document(document_file=os.path.join(args.data_dir, args.clinical_case+".txt"))
    doc_obj.parse_document()
    doc_obj.read_annotation(ann_file=os.path.join(args.data_dir, args.clinical_case+".ann"))
    doc_obj.parse_annotations()
    doc_obj.assign_ground_truth_ner_tags()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_case", action="store", dest="clinical_case")
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/MEDDOPROF/meddoprof-train-set/task1", dest="data_dir")
    parser.add_argument("--logging_level", action="store", default="INFO", dest="logging_level", help="options: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    args = parser.parse_args()

    main(args=args)
