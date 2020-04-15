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
from pymatgen.io.qchem.outputs import QCOutput

from pymatgen.io.qchem.utils import (lower_and_check_unique,
                                     read_pattern,
                                     read_table_pattern)

# Classes for reading/manipulating/writing output files for use with pyGSM.

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"
__email__ = "espottesmith@gmail.com"
__credit__ = "Sam Blau"

logger = logging.getLogger(__name__)


class GSMOutput(MSONable):

    """
    An object that parses and contains data from pyGSM output files.

    Args:
        filename (str): The path to the pyGSM output file.
    """

    def __init__(self, filename):
        self.filename = filename
        self.data = dict()
        self.data["errors"] = list()
        self.data["warnings"] = list()
        self.test = ""
        with zopen(filename, "rt") as f:
            self.text = f.read()

        completion = read_pattern(self.text, {"key": r"Finished GSM!"},
                                  terminate_on_match=True).get("key")
        if completion is None:
            self.data["completion"] = False
        else:
            self.data["completion"] = True

        self._parse_input_values()
        self._parse_initial_energies()
        self._parse_node_opt_trajectory()

        if self.data["inputs"]["gsm_type"] == "SE_GSM":
            self._parse_driving_coordinates()
            self._parse_coordinate_trajectory()

        self._parse_opt_summary()
        self._parse_summary_info()

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

    def _parse_initial_energies(self):

        if self.data["inputs"]["gsm_type"] == "SE_GSM":
            temp_init_energy = read_pattern(self.text,
                                            {"single": r"\s*Initial energy is ([\-\.0-9]+)"},
                                            terminate_on_match=True).get("single")
            self.data["initial_energy"] = float(temp_init_energy[0][0])

        elif self.data["inputs"]["gsm_type"] == "DE_GSM":
            temp_init_energy = read_pattern(self.text, {
                "double": r"\s*Energy of the end points are ([\-\.0-9]+), ([\-\.0-9]+)",
                "double_relative": r"\s*relative E ([\-\.0-9]+), ([\-\.0-9]+)"},
                                            terminate_on_match=True)
            if temp_init_energy.get("double"):
                self.data["initial_energy_rct"] = float(temp_init_energy.get("double")[0][0])
                self.data["initial_energy_pro"] = float(temp_init_energy.get("double")[0][1])
                self.data["initial_relative_energy_rct"] = float(temp_init_energy.get("double_relative")[0][0])
                self.data["initial_relative_energy_pro"] = float(temp_init_energy.get("double_relative")[0][1])

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
        #TODO: Hack pyGSM so that all of these return the indices associated with the coordinate
        #TODO: Also use consistent units?

        temp_coord_trajectory = read_pattern(self.text, {
            "add": r"\s*bond \(([0-9]+), ([0-9]+)\) target \(greater than\): ([\.0-9]+), current d: ([\.0-9]+) diff: [\-\.0-9]+",
            "break": r"s*bond \(([0-9]+), ([0-9]+)\) target \(greater than\): ([\.0-9]+), current d: ([\.0-9]+) diff: [\-\.0-9]+",
            "angle": r"s*anglev: ([\-\.0-9]+) align to ([\-\.0-9]+) diff(rad): [\-\.0-9]+",
            "torsion": r"s*current torv: ([\-\.0-9]+) align to ([\-\.0-9]+) diff(deg): [\-\.0-9]+"
        })

        # Initialize dictionary for driving coordinate trajectories
        self.data["driving_coord_trajectories"] = {"add": dict(),
                                                   "break": dict(),
                                                   "angle": dict(),
                                                   "torsion": dict()}

        self.data["driving_coord_goals"] = {"add": dict(),
                                            "break": dict(),
                                            "angle": dict(),
                                            "torsion": dict()}

        for coord_type, coords in self.data["driving_coords"].items():
            for coord in coords:
                self.data["driving_coord_trajectories"][coord_type][coord] = list()

        for add_coord in temp_coord_trajectory.get("add", list()):
            bond = (int(add_coord[0]), int(add_coord[1]))
            if bond in self.data["driving_coords"]["add"]:
                if bond not in self.data["driving_coord_goals"]["add"]:
                    self.data["driving_coord_goals"]["add"][bond] = float(add_coord[2])
                self.data["driving_coord_trajectories"]["add"][bond].append(float(add_coord[3]))

        for break_coord in temp_coord_trajectory.get("break", list()):
            bond = (int(break_coord[0]), int(break_coord[1]))
            if bond in self.data["driving_coords"]["break"]:
                if bond not in self.data["driving_coord_goals"]["break"]:
                    self.data["driving_coord_goals"]["break"][bond] = float(break_coord[2])
                self.data["driving_coord_trajectories"]["break"][bond].append(float(break_coord[3]))

        for e, ang_coord in enumerate(temp_coord_trajectory.get("angle", list())):
            #TODO: Fix this hack once the indices are printed for angles
            angle_ind = e % len(self.data["driving_coords"]["angle"])
            angle = self.data["driving_coords"]["angle"][angle_ind]
            if angle not in self.data["driving_coord_goals"]["angle"]:
                self.data["driving_coord_goals"]["angle"][angle] = float(ang_coord[1])
            self.data["driving_coord_trajectories"]["angle"][angle].append(float(ang_coord[0]))

        for e, tors_coord in enumerate(temp_coord_trajectory.get("torsion", list())):
            #TODO: Fix this hack once the indices are printed for angles
            tors_ind = e % len(self.data["driving_coords"]["torsion"])
            tors = self.data["driving_coords"]["torsion"][tors_ind]
            if tors not in self.data["driving_coord_goals"]["torsion"]:
                self.data["driving_coord_goals"]["torsion"][tors] = float(tors_coord[1])
            self.data["driving_coord_trajectories"]["torsion"][tors].append(float(tors_coord[0]))

    def _parse_node_opt_trajectory(self):
        self.data["opt_trajectory_energies"] = dict()
        self.data["opt_trajectory_gradrms"] = dict()

        header_pattern = r"\s*converged\n\s*opt-summary [0-9]+"
        body_pattern = r"\s*Node: ([0-9]+) Opt step: [0-9]+ E: ([\.\-0-9]+) predE: [\.\-0-9]+ ratio: [\.\-0-9]+ gradrms: ([\.0-9]+) ss: [\-\.0-9]+ DMAX: ([\.0-9]+)"
        footer_pattern = r""
        temp_opt_trajectories = read_table_pattern(self.text,
                                                   header_pattern=header_pattern,
                                                   row_pattern=body_pattern,
                                                   footer_pattern=footer_pattern)

        for table in temp_opt_trajectories:
            energies = list()
            grads = list()
            node_num = int(table[0][0])
            if node_num not in self.data["opt_trajectory_energies"]:
                self.data["opt_trajectory_energies"][node_num] = list()
            if node_num not in self.data["opt_trajectory_gradrms"]:
                self.data["opt_trajectory_gradrms"][node_num] = list()
            for row in table:
                energies.append(float(row[1]))
                grads.append(float(row[2]))

            self.data["opt_trajectory_energies"][node_num].append(energies)
            self.data["opt_trajectory_gradrms"][node_num].append(grads)

    def _parse_opt_summary(self):
        temp_opt_summary = read_pattern(self.text, {
            "v_profile": r"\s*V_profile:((\s+[\.\-0-9]+)+)",
            "v_profile_re": r"s*V_profile \(after reparam\):((\s+[\.\-0-9]+)+)",
            "all_uphill": r"\s*all uphill\?\s+(True|False)",
            "dissociative": r"\s*dissociative\?\s+(True|False)",
            "min_nodes": r"\s*min nodes\s+\[([0-9]+,? ?)+\]",
            "max_nodes": r"\s*max nodes\s+\[([0-9]+,? ?)+\]",
            "emax_nmax": r"\s*emax and nmax in find peaks ([\.\-0-9]+),([0-9]+)"
        })

        self.data["energy_profiles"] = list()
        self.data["reparameterized_energy_profiles"] = list()
        self.data["path_uphill"] = list()
        self.data["path_dissociative"] = list()
        self.data["path_min_nodes"] = list()
        self.data["path_max_nodes"] = list()
        self.data["max_nodes"] = list()
        self.data["max_energies"] = list()

        for v_profile in temp_opt_summary.get("v_profile", []):
            pass

    def _parse_warnings(self):
        pass

    def _parse_summary_info(self):
        pass

    def as_dict(self):
        pass
