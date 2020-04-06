# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import logging

import numpy as np

from monty.io import zopen
from monty.json import MSONable

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.qchem.inputs import QCInput

from pymatgen.io.qchem.utils import (lower_and_check_unique,
                                     read_pattern,
                                     read_table_pattern)

# Classes for reading/manipulating/writing input files for use with pyGSM.

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"
__email__ = "espottesmith@gmail.com"

logger = logging.getLogger(__name__)


class GSMOutput(MSONable):

    def __init__(self, filename):
        self.filename = filename
        self.data = dict()
        self.data["errors"] = list()
        self.data["warnings"] = list()
        self.test = ""
        with zopen(filename, "rt") as f:
            self.text = f.read()

        self._parse_input_values()

        if self.data["inputs"]["gsm_type"] in ["SE_GSM", "SE_Cross"]:
            self._parse_driving_coordinates()
            self._parse_coordinate_trajectory()

    def _parse_input_values(self):
        header_pattern = r"#=+#\n#\|.+\[92m\s+Parsed GSM Keys : Values.+\[0m\s+\|#\n#=+#"
        table_pattern = r"(?P<key>[A-Za-z_]+)\s+(?P<value>[A-Za-z0-9\[\]_\.\-]+)\s*\n"
        footer_pattern = r"\-+"

        temp_inputs = read_table_pattern(self.text, header_pattern,
                                         table_pattern, footer_pattern)

        if temp_inputs is None or len(temp_inputs) == 0:
            self.data["inputs"] = dict()
        else:
            self.data["inputs"] = dict()
            temp_inputs = temp_inputs[0]

            for row in temp_inputs:
                key = row["key"]
                value = row["value"]

                if value == "True":  # Deal with bools
                    self.data["inputs"][key] = True
                elif value == "False":
                    self.data["inputs"][key] = False
                elif value.startswith("[") and value.endswith("]"):  # Deal with lists
                    val = value[1:-1].split(", ")
                    try:  # ints
                        val = [int(v) for v in val]
                        self.data["inputs"][key] = val
                    except ValueError:
                        self.data["inputs"][key] = val
                else:
                    # int
                    is_int = True
                    is_float = True
                    val = value
                    try:
                        val = int(value)
                    except ValueError:
                        is_int = False
                    if is_int:
                        self.data["inputs"][key] = val
                        continue
                    else:
                        try:
                            val = float(value)
                        except ValueError:
                            is_float = False

                    if is_float:
                        self.data["inputs"][key] = val
                        continue
                    else:
                        self.data["inputs"][key] = value

        if "charge" not in self.data["inputs"]:
            self.data["inputs"]["charge"] = 0

    def _parse_driving_coordinates(self):
        temp_coords = read_pattern(self.text, {
            "key": r"driving coordinates \[((\['(ADD|BREAK|ANGLE|TORSION|OOP)', ([0-9]+,? ?)+\],? ?)+)\]"
        }, terminate_on_match=True).get("key")

        self.data["driving_coords"] = dict()
        self.data["driving_coords"]["add"] = list()
        self.data["driving_coords"]["break"] = list()
        self.data["driving_coords"]["angle"] = list()
        self.data["driving_coords"]["torsion"] = list()
        self.data["driving_coords"]["out_of_plane"] = list()

        coord_sets = temp_coords[0][0].split("], [")
        for coord_set in coord_sets:
            tokens = coord_set.strip("[]").split(", ")
            self.data["driving_coords"][tokens[0].strip("'").lower()].append(tuple([int(e) for e in tokens[1:]]))

    def _parse_coordinate_trajectory(self):
        pass

    def _parse_node_opt_trajectory(self):
        pass

    def _parse_opt_summary(self):
        pass

    def _parse_warnings(self):
        pass

    def _parse_summary_info(self):
        pass

    def as_dict(self):
        pass
