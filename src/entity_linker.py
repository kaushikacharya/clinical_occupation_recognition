#!/usr/bin/env python

"""
References
----------
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html

"""

import argparse
from csv import writer
try:
    import fasttext
except ImportError:
    print("Failed to load fasttext")
import io
import logging
import numpy as np
import os
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

from src.annotation import *
from src.document import Document
from src.utils import *


class EntityLinker:
    def __init__(self, logging_level=logging.INFO):
        self.text_to_concept_identifier_map = dict()
        self.concept_identifier_to_text_map = dict()
        self.fasttext_model = None

        # logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging_level)

    def load_fasttext_model(self, filename):
        self.fasttext_model = fasttext.load_model(filename)

    @staticmethod
    def load_concept_identifiers(filename):
        df = pd.read_csv(filepath_or_buffer=filename, delimiter="\t")
        return df

    @staticmethod
    def process_concept_knowledge_base(df):
        """Process concept knowledge base.
            Includes
            a) Combining label and alternative_label.
            b) Splitting labels based on delimiters.
        """
        csv_output = io.StringIO()
        csv_writer = writer(csv_output)

        for i in range(len(df)):
            concept_labels = []
            if not pd.isnull(df.loc[i, "label"]):
                labels = re.split(r"[/|]", df.loc[i, "label"])
                concept_labels.extend(labels)

            if not pd.isnull(df.loc[i, "alternative_label"]):
                labels = re.split(r"[/|]", df.loc[i, "alternative_label"])
                concept_labels.extend(labels)

            for concept_label in concept_labels:
                if concept_label == "":
                    # case: Both the delimiters exist continuously
                    #       Example: "alternative_label" for code: "1324.3.2.1"
                    continue
                csv_row_data = [concept_label, df.loc[i, "code"]]
                csv_writer.writerow(csv_row_data)

        # Populate the processed dataframe
        csv_output.seek(0)
        df_processed = pd.read_csv(filepath_or_buffer=csv_output, names=["text", "code"])

        return df_processed

    def build_concept_identifier_map(self, df):
        """Build concept identifier map.

            Note
            ----
            User need to ensure that `df` contains unique text.
        """
        for idx in df.index:
            text = df.loc[idx, "text"]
            concept_identifier = df.loc[idx, "code"]

            if pd.isnull(df.loc[idx, "text"]):
                self.logger.warning("text empty for idx: {}".format(idx))
                continue

            if pd.isnull(df.loc[idx, "code"]):
                self.logger.warning("code empty for idx: {}".format(idx))
                continue

            if concept_identifier not in self.concept_identifier_to_text_map:
                self.concept_identifier_to_text_map[concept_identifier] = []

            self.concept_identifier_to_text_map[concept_identifier].append(text)

            if text in self.text_to_concept_identifier_map:
                self.logger.warning("text: {} :: concept identifier: {} :: already mapped to concept identifier: {}".format(
                    text, concept_identifier, self.text_to_concept_identifier_map[text]))
                continue

            self.text_to_concept_identifier_map[text] = concept_identifier

    def populate_ground_truth_embedding(self):
        csv_output = io.StringIO()
        csv_writer = writer(csv_output)

    def read_predicted_annotations(self, clinical_cases, text_data_dir, ann_data_dir):
        csv_output = io.StringIO()
        csv_writer = writer(csv_output)

        for clinical_case in clinical_cases:
            file_doc = os.path.join(text_data_dir, clinical_case + ".txt")
            file_ann = os.path.join(ann_data_dir, clinical_case + ".ann")

            if not os.path.exists(file_doc):
                self.logger.info("Text file not available for clinical case: {}".format(clinical_case))
                continue

            if not os.path.exists(file_ann):
                self.logger.info("Annotation file not available for clinical case: {}".format(clinical_case))
                continue

            try:
                doc_obj = Document(doc_case=clinical_case)
                doc_obj.read_document(document_file=file_doc)
                doc_entity_annotations = read_annotation(ann_file=file_ann)

                for entity_ann in doc_entity_annotations:
                    start_char_pos_ann = entity_ann.start_char_pos
                    end_char_pos_ann = entity_ann.end_char_pos
                    entity_text = doc_obj.text[start_char_pos_ann: end_char_pos_ann]
                    csv_row_data = [clinical_case, entity_text, " ".join([str(start_char_pos_ann), str(end_char_pos_ann)])]
                    csv_writer.writerow(csv_row_data)
            except Exception as err:
                self.logger.error("Failed for clinical case: {}".format(clinical_case), exc_info=True)

        # Now populate the dataframe
        csv_output.seek(0)
        df = pd.read_csv(filepath_or_buffer=csv_output, names=["filename", "text", "span"])
        self.logger.info("Predicted entities extracted in ann_data_dir : {}".format(len(df)))

        return df

    def assign_concept_identifier(self, df_concept_identifier, df_predicted_ann):
        # Create a new dataframe with unique `text`. Multiple `filename` can have common `text`.
        df_predicted_ann_unique = pd.DataFrame(data={"text": df_predicted_ann["text"].unique()})
        # Add empty column: code. This will be populated based on cosine similarity of vector embedding.
        df_predicted_ann_unique = df_predicted_ann_unique.reindex(columns=df_predicted_ann_unique.columns.tolist()+["code"])

        # Extract vector embedding for the text
        concept_identifier_vec_arr = np.array([self.fasttext_model.get_sentence_vector(x) for x in df_concept_identifier["text"]])
        predicted_vec_arr = np.array([self.fasttext_model.get_sentence_vector(x) for x in df_predicted_ann_unique["text"]])

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(X=predicted_vec_arr, Y=concept_identifier_vec_arr, dense_output=True)
        assert similarity_matrix.shape == (len(predicted_vec_arr), len(concept_identifier_vec_arr)),\
            "Mismatch shape: {} :: len(predicted_vec_arr): {} :: len(concept_identifier_vec_arr): {}".format(
                similarity_matrix.shape, len(predicted_vec_arr), len(concept_identifier_vec_arr))

        best_similarity_index_arr = similarity_matrix.argmax(axis=1)
        assert best_similarity_index_arr.shape == (len(predicted_vec_arr), ),\
            "Mismatch: best_similarity_index_arr.shape: {} :: len(predicted_vec_arr): {}".format(
                best_similarity_index_arr.shape, len(predicted_vec_arr))

        # Assign concept identifier based on the one which has highest cosine similarity
        for i in range(len(best_similarity_index_arr)):
            chosen_code = df_concept_identifier.loc[best_similarity_index_arr[i], "code"]
            df_predicted_ann_unique.loc[i, "code"] = chosen_code

        # Populate `code` of predicted entity dataframe by merging
        df_predicted_ann = pd.merge(left=df_predicted_ann, right=df_predicted_ann_unique, on="text")

        return df_predicted_ann


def main(args):
    entity_linker_obj = EntityLinker()
    df_concept_identifier = entity_linker_obj.load_concept_identifiers(filename=args.concept_identifier_file)
    # Keep only the text and concept identifier columns
    df_concept_identifier = df_concept_identifier[["text", "code"]]
    # Load the concept knowledge base
    df_concept_knowledge_base = entity_linker_obj.load_concept_identifiers(filename=args.concept_knowledge_base_file)
    # Process the concept knowledge base by splitting text
    df_concept_knowledge_base_processed = entity_linker_obj.process_concept_knowledge_base(df=df_concept_knowledge_base)
    # Now stack the two dataframes
    df_concept_identifier = pd.concat([df_concept_identifier, df_concept_knowledge_base_processed], axis=0, ignore_index=True)
    # Since same entity text could be present in multiple clinical cases, remove the duplicates as we are now only
    # concerned with text and its corresponding concept identifier.
    df_concept_identifier.drop_duplicates(inplace=True)
    entity_linker_obj.build_concept_identifier_map(df=df_concept_identifier)
    clinical_cases = collect_clinical_cases(data_dir=args.ann_data_dir, file_extension="ann")
    df_predicted_ann = entity_linker_obj.read_predicted_annotations(clinical_cases=clinical_cases, text_data_dir=args.text_data_dir, ann_data_dir=args.ann_data_dir)

    df_concept_identifier.to_csv(path_or_buf=os.path.join(os.path.dirname(__file__), "../output/entity_linker", "concept_identifier.csv"), index=False, encoding="utf-8")
    if False:
        with open(os.path.join(os.path.dirname(__file__), "../output/entity_linker", "concept_identifier.csv"), mode="w") as fd:
            df_concept_identifier.to_csv(path_or_buf=fd, index=False)

    df_predicted_ann.to_csv(path_or_buf=os.path.join(os.path.dirname(__file__), "../output/entity_linker", "predicted_ann.csv"), index=False, encoding="utf-8")
    if False:
        with open(os.path.join(os.path.dirname(__file__), "../output/entity_linker", "predicted_ann.csv"), mode="w") as fd:
            df_predicted_ann.to_csv(path_or_buf=fd, index=False, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_identifier_file", action="store",
                        default="C:/KA/data/NLP/MEDDOPROF/meddoprof-train-set/meddoprof-norm-train.tsv", dest="concept_identifier_file")
    parser.add_argument("--concept_knowledge_base_file", action="store", default="C:/KA/lib/meddoprof-evaluation-library/meddoprof_valid_codes.tsv", dest="concept_knowledge_base_file")
    parser.add_argument("--text_data_dir", action="store", default="C:/KA/data/NLP/MEDDOPROF/meddoprof_test_txt/",
                        dest="text_data_dir")
    parser.add_argument("--ann_data_dir", action="store", default=os.path.join(os.path.dirname(__file__), "../output/predict/meddoprof_test_txt"),
                        dest="ann_data_dir")
    args = parser.parse_args()

    main(args=args)
