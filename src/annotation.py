#!/usr/bin/env python

import io
import logging
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EntityAnnotation:
    """Entity annotation class"""
    def __init__(self, entity_id=None, start_char_pos=None, end_char_pos=None, start_token_index=None,
                 end_token_index=None, sentence_index=None, entity_type=None):
        """
            References
            ----------
            https://brat.nlplab.org/standoff.html
        """
        # entity_id represents the numeral that follows T in .ann file.
        self.id = entity_id
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos
        # token index range: [start_token_index, end_token_index)
        #   N.B. start_token_index is included but end_token_index is excluded.
        self.start_token_index = start_token_index
        self.end_token_index = end_token_index
        self.sentence_index = sentence_index
        self.type = entity_type


def read_annotation(ann_file):
    """
    Read brat standoff formatted annotation file.

    Returns
    -------
    entity_annotations : list (list of EntityAnnotation in a Document)

    References
    ----------
    https://brat.nlplab.org/standoff.html
    """
    entity_annotations = []

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
            entity_annotations.append(entity_ann)

    # Sort wrt start_char_pos
    entity_annotations = sorted(entity_annotations, key=lambda x: x.start_char_pos)

    return entity_annotations


def parse_annotations(entity_annotations, doc_obj):
    """Extract and assign token range and sentence index of corresponding document to the entity annotations.

        Returns
        -------
        None  (Since list (entity_annotations) is a mutable object, hence no need to return)
    """
    entity_ann_index = 0
    sent_i = 0

    while (sent_i < len(doc_obj.sentences)) and (entity_ann_index < len(entity_annotations)):
        start_char_pos_sent = doc_obj.sentences[sent_i].start_char_pos
        end_char_pos_sent = doc_obj.sentences[sent_i].end_char_pos
        start_char_pos_ann = entity_annotations[entity_ann_index].start_char_pos
        end_char_pos_ann = entity_annotations[entity_ann_index].end_char_pos

        if end_char_pos_sent <= start_char_pos_ann:
            # move to the next sentence
            sent_i += 1
            continue
        elif start_char_pos_sent <= start_char_pos_ann < end_char_pos_sent:
            # Current entity belongs to the current sentence
            # Identify token range for the current entity
            start_token_index_ann = None
            end_token_index_ann = None

            token_index = doc_obj.sentences[sent_i].start_token_index
            while token_index < doc_obj.sentences[sent_i].end_token_index:
                start_char_pos_token = doc_obj.tokens[token_index].start_char_pos
                end_char_pos_token = doc_obj.tokens[token_index].end_char_pos

                if start_char_pos_token <= start_char_pos_ann < end_char_pos_token:
                    start_token_index_ann = token_index
                    break

                # increment token index
                token_index += 1

            assert start_token_index_ann is not None, "start_token_index_ann not assigned. Sentence #{}".format(sent_i)
            while token_index < doc_obj.sentences[sent_i].end_token_index:
                start_char_pos_token = doc_obj.tokens[token_index].start_char_pos
                end_char_pos_token = doc_obj.tokens[token_index].end_char_pos

                if end_char_pos_ann > end_char_pos_token:
                    # move to the next token
                    token_index += 1
                else:
                    end_token_index_ann = token_index + 1
                    break

            if end_token_index_ann is None:
                end_token_index_ann = doc_obj.sentences[sent_i].end_token_index

            assert entity_annotations[entity_ann_index].start_token_index is None,\
                "start_token_index already assigned for entity #{}".format(entity_ann_index)
            assert entity_annotations[entity_ann_index].end_token_index is None,\
                "end_token_index already assigned for entity #{}".format(entity_ann_index)

            # Assign the token index range to the entity annotation
            entity_annotations[entity_ann_index].start_token_index = start_token_index_ann
            entity_annotations[entity_ann_index].end_token_index = end_token_index_ann

            # Assign sentence index
            entity_annotations[entity_ann_index].sentence_index = sent_i

            # move to the next entity
            entity_ann_index += 1
        else:
            assert False, "entity_ann_index: {} missed assignment of token range".format(entity_ann_index)


def compare_annotations(entity_annotations_truth, entity_annotations_predicted, doc_obj):
    """Compare predicted entities with ground truth entities' text span to find which
        a) match exactly
        b) partially match
        c) no match

        No match are further categorized as
        a) False positive (Additional predicted entity generated)
        b) False negative (No predicted entity for a ground truth entity)
    """
    ann_truth_i = 0
    ann_pred_i = 0
    sent_i = 0

    while (ann_truth_i < len(entity_annotations_truth)) and (ann_pred_i < len(entity_annotations_predicted)):
        entity_ann_truth = entity_annotations_truth[ann_truth_i]
        entity_ann_pred = entity_annotations_predicted[ann_pred_i]

        if entity_ann_pred.end_char_pos <= entity_ann_truth.start_char_pos:
            flag_false_positive = False
            if ann_truth_i > 0:
                # Check whether previous truth entity overlaps with the current predicted entity.
                entity_ann_truth_prev = entity_annotations_truth[ann_truth_i-1]
                if entity_ann_truth_prev.end_char_pos <= entity_ann_pred.start_char_pos:
                    # case: False positive
                    flag_false_positive = True
            else:
                # case: False positive
                # Since there's no truth entity prior to the current truth entity, we can conclude that the
                # predicted entity is a false positive.
                flag_false_positive = True

            if flag_false_positive:
                while doc_obj.sentences[sent_i].end_char_pos <= entity_ann_pred.start_char_pos:
                    sent_i += 1
                logger.info("False Positive: Type: {} :: Predicted entity #{} :: char pos range: ({},{}) :: entity text: {} :: "
                            "sentence: {}"
                            .format(entity_ann_pred.type, ann_pred_i, entity_ann_pred.start_char_pos, entity_ann_pred.end_char_pos,
                                    doc_obj.text[entity_ann_pred.start_char_pos: entity_ann_pred.end_char_pos],
                                    doc_obj.text[doc_obj.sentences[sent_i].start_char_pos: doc_obj.sentences[sent_i].end_char_pos]))
        elif entity_ann_truth.end_char_pos <= entity_ann_pred.start_char_pos:
            # Current ground truth entity can be considered as false negative if it doesn't overlap with
            # previous predicted entity
            flag_false_negative = False
            if ann_pred_i > 0:
                # Check whether the previous predicted entity overlaps with the current truth entity.
                entity_ann_pred_prev = entity_annotations_predicted[ann_pred_i-1]
                if entity_ann_pred_prev.end_char_pos <= entity_ann_truth.start_char_pos:
                    # case: False negative
                    flag_false_negative = True
            else:
                # case: False negative
                # Since there's no predicted entity prior to the current predicted entity, hence we can conclude
                # that the  current truth entity was not generated by the model.
                flag_false_negative = True

            if flag_false_negative:
                while doc_obj.sentences[sent_i].end_char_pos <= entity_ann_truth.start_char_pos:
                    sent_i += 1
                logger.info("False Negative: Type: {} :: Truth entity #{} :: char pos range: ({},{}) :: entity text: {} :: "
                            "sentence: {}"
                            .format(entity_ann_truth.type, ann_truth_i, entity_ann_truth.start_char_pos, entity_ann_truth.end_char_pos,
                                    doc_obj.text[entity_ann_truth.start_char_pos: entity_ann_truth.end_char_pos],
                                    doc_obj.text[doc_obj.sentences[sent_i].start_char_pos: doc_obj.sentences[sent_i].end_char_pos]))

        elif (entity_ann_pred.start_char_pos == entity_ann_truth.start_char_pos) and\
                (entity_ann_pred.end_char_pos == entity_ann_truth.end_char_pos):
            # case: Exact match
            while doc_obj.sentences[sent_i].end_char_pos <= entity_ann_truth.start_char_pos:
                sent_i += 1
            logger.info("Exact Match: Truth entity #{} :: Predicted entity #{} :: char pos range: ({},{}) ::"
                        " entity text: {} :: sentence: {}"
                        .format(ann_truth_i, ann_pred_i, entity_ann_truth.start_char_pos,entity_ann_truth.end_char_pos,
                                doc_obj.text[entity_ann_truth.start_char_pos: entity_ann_truth.end_char_pos],
                                doc_obj.text[doc_obj.sentences[sent_i].start_char_pos: doc_obj.sentences[sent_i].end_char_pos]))
        else:
            # case: Partial match
            while doc_obj.sentences[sent_i].end_char_pos <= entity_ann_truth.start_char_pos:
                sent_i += 1
            logger.info("Partial Match: Truth: entity #{} :: Predicted entity #{} :: char pos range(truth): ({},{}) ::"
                        " entity text(truth): {} :: char pos range(predicted): ({},{}) :: entity text(predicted): {} ::"
                        " sentence: {}"
                        .format(ann_truth_i, ann_pred_i, entity_ann_truth.start_char_pos, entity_ann_truth.end_char_pos,
                                doc_obj.text[entity_ann_truth.start_char_pos: entity_ann_truth.end_char_pos],
                                entity_ann_pred.start_char_pos, entity_ann_pred.end_char_pos,
                                doc_obj.text[entity_ann_pred.start_char_pos: entity_ann_pred.end_char_pos],
                                doc_obj.text[doc_obj.sentences[sent_i].start_char_pos: doc_obj.sentences[sent_i].end_char_pos]))

        if entity_ann_pred.end_char_pos == entity_ann_truth.end_char_pos:
            # For next iteration, move to the next predicted and truth entity
            ann_pred_i += 1
            ann_truth_i += 1
        elif entity_ann_pred.end_char_pos < entity_ann_truth.end_char_pos:
            # For next iteration, move to the next predicted entity only.
            #   Keep the truth entity pointer as it is, since next predicted entity needs to be compared with the
            #   current truth entity.
            ann_pred_i += 1
        else:
            ann_truth_i += 1

    if ann_truth_i < len(entity_annotations_truth):
        # First check if the current truth entity partial matches with the last predicted entity
        #   If it partially matches, then that would have been already marked as partial match.
        if len(entity_annotations_predicted) > 0:
            entity_ann_pred = entity_annotations_predicted[-1]
            entity_ann_truth = entity_annotations_truth[ann_truth_i]

            if entity_ann_truth.start_char_pos >= entity_ann_pred.end_char_pos:
                # case: False negative
                while doc_obj.sentences[sent_i].end_char_pos <= entity_ann_truth.start_char_pos:
                    sent_i += 1
                logger.info("False Negative: Type: {} :: Truth entity #{} :: char pos range: ({},{}) :: entity text: {} :: "
                            "sentence: {}"
                            .format(entity_ann_truth.type, ann_truth_i, entity_ann_truth.start_char_pos, entity_ann_truth.end_char_pos,
                                    doc_obj.text[entity_ann_truth.start_char_pos: entity_ann_truth.end_char_pos],
                                    doc_obj.text[doc_obj.sentences[sent_i].start_char_pos: doc_obj.sentences[sent_i].end_char_pos]))

            ann_truth_i += 1

        # rest of the truth entity can be safely assigned as false negative
        while ann_truth_i < len(entity_annotations_truth):
            # case: False negative
            entity_ann_truth = entity_annotations_truth[ann_truth_i]
            while doc_obj.sentences[sent_i].end_char_pos <= entity_ann_truth.start_char_pos:
                sent_i += 1
            logger.info("False Negative: Type: {} :: Truth entity #{} :: char pos range: ({},{}) :: entity text: {} :: sentence: {}"
                        .format(entity_ann_truth.type, ann_truth_i, entity_ann_truth.start_char_pos, entity_ann_truth.end_char_pos,
                                doc_obj.text[entity_ann_truth.start_char_pos: entity_ann_truth.end_char_pos],
                                doc_obj.text[doc_obj.sentences[sent_i].start_char_pos: doc_obj.sentences[sent_i].end_char_pos]))
            ann_truth_i += 1

    if ann_pred_i < len(entity_annotations_predicted):
        # First check if the current predicted entity matches with the last truth entity
        if len(entity_annotations_truth) > 0:
            entity_ann_truth = entity_annotations_truth[-1]
            entity_ann_pred = entity_annotations_predicted[ann_pred_i]

            if entity_ann_pred.start_char_pos >= entity_ann_truth.end_char_pos:
                # case: False positive
                while doc_obj.sentences[sent_i].end_char_pos <= entity_ann_pred.start_char_pos:
                    sent_i += 1
                logger.info("False Positive: Type: {} :: Predicted entity #{} :: char pos range: ({},{}) :: "
                            "entity text: {} :: sentence: {}"
                            .format(entity_ann_pred.type, ann_pred_i, entity_ann_pred.start_char_pos, entity_ann_pred.end_char_pos,
                                    doc_obj.text[entity_ann_pred.start_char_pos: entity_ann_pred.end_char_pos],
                                    doc_obj.text[doc_obj.sentences[sent_i].start_char_pos: doc_obj.sentences[sent_i].end_char_pos]))

            # Move to the next truth entity
            ann_pred_i += 1

        # rest of the predicted entity can be safely assigned as false positive
        while ann_pred_i < len(entity_annotations_predicted):
            # case: False positive
            entity_ann_pred = entity_annotations_predicted[ann_pred_i]
            while doc_obj.sentences[sent_i].end_char_pos <= entity_ann_pred.start_char_pos:
                sent_i += 1
            logger.info("False Positive: Type: {} :: Predicted entity #{} :: char pos range: ({},{}) :: "
                        "entity text: {} :: sentence: {}"
                        .format(entity_ann_pred.type, ann_pred_i, entity_ann_pred.start_char_pos, entity_ann_pred.end_char_pos,
                                doc_obj.text[entity_ann_pred.start_char_pos: entity_ann_pred.end_char_pos],
                                doc_obj.text[doc_obj.sentences[sent_i].start_char_pos: doc_obj.sentences[sent_i].end_char_pos]))
            # Move to the next predicted entity
            ann_pred_i += 1
