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


class XTBOutput(MSONable):
    """
    Class to parse XTB outputs
    """

    def __init__(self, path, output_filename, namespace=None):
        """
        Args:
            path (str): Path to directory including output_filename and all
                other xtb output files (xtbopt.log, vibspectrum, etc.)
            output_filename (str): Filename to parse
            namespace (str): If the namespace is not None (default), this will
                be prepended to all filenames other than output_filename
        """

        self.filename = output_filename
        self.namespace = namespace
        self.data = dict()

        self.data["errors"] = list()
        self.data["warnings"] = dict()

        self.text = ""
        with zopen(output_filename, "rt") as f:
            self.text = f.read()

        self._read_parameters()

        # Need some condition to see if this is an opt job

    def _read_parameters(self):



def parse_xtb_output(file_path="xtb.out"):
    """
    Things we need to parse:
    - Final energy
    - Final enthalpy
    - Final entropy
    - Final free energy
    - Heat capacity
    - Charge?

    :param file_path:
    :return:
    """

    # In Hartree
    energy_pattern = re.compile(r"\s+\| TOTAL ENERGY\s+(?P<energy>[\-\.0-9]+) Eh")

    # In hartree
    tot_enthalpy = re.compile(r"\s+\| TOTAL ENTHALPY\s+(?P<total_enthalpy>[\-\.0-9]+) Eh")

    # In hartree
    tot_gibbs = re.compile(r"\s+\| TOTAL FREE ENERGY\s+(?P<total_free_energy>[\-\.0-9]+) Eh")

    # In cal/mol, cal/mol-K, cal/mol-K, respectively
    tot_pattern = re.compile(r"TOT\s+(?P<enthalpy>[\-\.0-9]+)\s+(?P<heat_capacity>[\-\.0-9]+)\s+(?P<entropy>[\-\.0-9]+)")

    with open(file_path, 'r') as xtbout_file:
        contents = xtbout_file.read()

        total_energy = float(energy_pattern.search(contents).group("energy"))
        total_enthalpy = float(tot_enthalpy.search(contents).group("total_enthalpy"))
        total_gibbs = float(tot_gibbs.search(contents).group("total_free_energy"))

        tot_match = tot_pattern.search(contents)
        enthalpy = float(tot_match.group("enthalpy")) / 1000
        heat_capacity = float(tot_match.group("heat_capacity"))
        entropy = float(tot_match.group("entropy"))

        return {"energy": total_energy,
                "total_h": total_enthalpy,
                "total_g": total_gibbs,
                "enthalpy": enthalpy,
                "cp": heat_capacity,
                "entropy": entropy}