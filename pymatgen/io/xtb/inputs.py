# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import logging
import numpy as np
from monty.json import MSONable
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from .utils import (read_table_pattern,
                    read_pattern,
                    lower_and_check_unique,
                    generate_string_start)

# Classes for reading/manipulating/writing QChem input files.

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"

logger = logging.getLogger(__name__)


class XTBInput(MSONable):
