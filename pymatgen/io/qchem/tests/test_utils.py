# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import os
from os.path import join
import unittest

import numpy as np

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import CovalentBondNN, OpenBabelNN
from pymatgen.io.qchem.utils import (map_atoms_reaction,
                                     orient_molecule,
                                     generate_string_start)
# from pymatgen.util.testing import PymatgenTest

try:
    import openbabel
    have_babel = True
except ImportError:
    have_babel = False

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"

test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        'test_files', "molecules")


class QCUtilsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.rct_1 = Molecule.from_file(join(test_dir, "da_reactant_1.mol"))
        self.rct_2 = Molecule.from_file(join(test_dir, "da_reactant_2.mol"))
        self.pro = Molecule.from_file(join(test_dir, "da_product.mol"))

    def tearDown(self) -> None:
        del self.rct_1
        del self.rct_2
        del self.pro

if __name__ == "__main__":
    unittest.main()