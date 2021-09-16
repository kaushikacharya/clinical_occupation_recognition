#!/usr/bin/env python

import argparse
import io
import logging
import spacy
from spacy import displacy

"""
References
----------
- https://spacy.io/usage/visualizers#dep-long-text
    - Renders dependency parse tree sentence by sentence.
"""


class NLPProcess:
    """NLP processing of text
    """
    def __init__(self, model="es_core_news_sm", logging_level=logging.INFO):
        self.model = model
        self.nlp = None
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

    def load_nlp_model(self):
        self.nlp = spacy.load(name=self.model)
        self.logger.info("Model: {} loaded".format(self.model))
        self.logger.info("pipe names: {}".format(self.nlp.pipe_names))

    def construct_doc(self, text):
        """Construct Doc container from the text.

            Reference:
            ---------
            https://spacy.io/api/doc
        """
        assert self.nlp is not None, "pre-requisite: Execute load_nlp_model()"
        doc = self.nlp(text)
        return doc


def main(args, verbose=True):
    import os

    obj_nlp_process = NLPProcess()
    obj_nlp_process.load_nlp_model()

    file_document = os.path.join(args.data_dir, args.clinical_case + ".txt")

    output_dir = os.path.join(os.path.dirname(__file__), "../output/debug", args.clinical_case)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with io.open(file=file_document, encoding="utf-8") as fd_doc:
        file_text = fd_doc.read()
        # print(file_text)

        doc = obj_nlp_process.construct_doc(text=file_text)

        if verbose:
            # svg = displacy.render(doc, style="dep")
            for sent_i, sent in enumerate(doc.sents):
                print("Sentence #{}: {}".format(sent_i, sent.text))
                svg = displacy.render(sent, style="dep")
                svg_sentence_file = os.path.join(output_dir, str(sent_i)+".svg")
                with io.open(file=svg_sentence_file, mode="w", encoding="utf-8") as fd_svg:
                    fd_svg.write(svg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_case", action="store", dest="clinical_case")
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/MEDDOPROF/meddoprof-train-set/task1", dest="data_dir")

    args = parser.parse_args()

    main(args=args)
