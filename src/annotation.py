#!/usr/bin/env python

class EntityAnnotation:
    """Entity annotation class"""
    def __init__(self, entity_id=None, start_char_pos=None, end_char_pos=None, start_token_index=None, end_token_index=None, sentence_index=None, entity_type=None):
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
