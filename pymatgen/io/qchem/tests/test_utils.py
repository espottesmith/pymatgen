# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import os
import unittest

import numpy as np

from pymatgen.io.qchem.utils import (lower_and_check_unique,
                                     process_parsed_coords)
from pymatgen.util.testing import PymatgenTest

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

mol_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "test_files",
    "molecules")

qchem_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "test_files",
    "qchem")


class QChemUtilsTest(PymatgenTest):
    def test_lower_and_check_unique(self):
        no_change_dict = {'a': True,
                          'b': "bee",
                          'c': 1234}

        change_dict = {'A': True,
                       'B': "Bee",
                       'C': 1234}

        bad_dict = {'A': True,
                    'a': False}

        self.assertDictEqual(no_change_dict,
                             lower_and_check_unique(no_change_dict))

        self.assertDictEqual(no_change_dict,
                             lower_and_check_unique(change_dict))

        with self.assertRaises(Exception):
            lower_and_check_unique(bad_dict)

    def test_process_parsed_coords(self):
        good_string = [["1.1", "2.2", "3.3"],
                       ["0.0", "0.0", "0.0"]]

        bad_string = [["1.1", "2.2"]]

        array = np.array([[1.1, 2.2, 3.3], [0.0, 0.0, 0.0]])
        parsed = process_parsed_coords(good_string)
        for i in range(len(good_string)):
            for j in range(3):
                self.assertEqual(parsed[i][j], array[i][j])

        with self.assertRaises(ValueError):
            process_parsed_coords(bad_string)


if __name__ == '__main__':
    unittest.main()
