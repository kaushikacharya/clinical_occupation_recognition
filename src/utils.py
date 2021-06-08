#!/usr/bin/env python

import glob
import os


def collect_clinical_cases(data_dir):
    """Collect clinical cases and sort to ensure consistent order.

        Parameters
        ----------
        data_dir : Path

        References
        ----------
        https://docs.python.org/3.7/library/glob.html
            - Python 3 uses os.scandir(), whereas python 2 uses os.listdir()
            - "results are returned in arbitrary order"
        https://docs.python.org/3.7/library/os.html#os.scandir
            - "The entries are yielded in arbitrary order"
    """
    clinical_cases = []
    for f in glob.iglob(pathname=os.path.join(data_dir, "*.txt")):
        # extract clinical case
        file_basename, _ = os.path.splitext(os.path.basename(f))
        clinical_cases.append(file_basename)

    # Since as per the documentation, os.scandir() yields entries in arbitrary order, sorting to ensure consistency.
    clinical_cases = sorted(clinical_cases)

    return clinical_cases
