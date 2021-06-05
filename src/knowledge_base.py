#!/usr/bin/env python

import argparse
import io


class Gazetteer:
    def __init__(self):
        # trie with each word nodes
        self.gazetteer_phrase_trie = dict()

    def build_gazetteer_phrase_trie(self, filename, delimiter="\t"):
        with io.open(filename, mode="r", encoding="utf-8") as fd:
            for line in fd:
                line = line.strip()
                if line == "":
                    continue
                phrases = line.split(sep=delimiter)
                # use the first phrase
                tokens = phrases[0].split()
                cur_node = self.gazetteer_phrase_trie
                for token in tokens:
                    if token.istitle():
                        token = token.lower()
                    if token not in cur_node:
                        cur_node[token] = dict()
                    cur_node = cur_node[token]

                # End of phrase
                cur_node["EOP"] = True
                # print("line: {}".format(phrases[0]))

    def search(self, tokens):
        """Search for the input phrase (split into tokens) in the phrase trie.

            Parameters
            ----------
            tokens : list (split tokens of the phrase to be searched)
        """
        cur_node = self.gazetteer_phrase_trie
        for token in tokens:
            if token.istitle():
                token = token.lower()
            if token in cur_node:
                cur_node = cur_node[token]
            else:
                return False

        # consider found only if the path end reached
        return True if "EOP" in cur_node else False


def main(args):
    gazetteer_obj = Gazetteer()
    gazetteer_obj.build_gazetteer_phrase_trie(filename=args.gazetteer_file)
    # flag_found = gazetteer_obj.search("agentes de Salud Comunitaria".split())
    # print("Found: {}".format(flag_found))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gazetteer_file", action="store", default="C:/KA/data/NLP/MEDDOPROF/occupations-gazetteer/profner-gazetteer.tsv", dest="gazetteer_file")
    args = parser.parse_args()

    main(args=args)
