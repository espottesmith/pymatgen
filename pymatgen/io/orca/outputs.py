"""
Parsers for ORCA output files.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import re
import warnings
from typing import Any, Dict

import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize

from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.core.structure import Molecule
from pymatgen.io.qchem.outputs import nbo_parser
from pymatgen.io.qchem.utils import (
    process_parsed_coords,
    process_parsed_fock_matrix,
    read_matrix_pattern,
    read_pattern,
    read_table_pattern,
)

try:
    from openbabel import openbabel
except ImportError:
    openbabel = None


__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2023, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__credits__ = "Samuel Blau, Gabe Gomes"

logger = logging.getLogger(__name__)


class ORCAOutput(MSONable):
    """
    Class to parse ORCA output files (typically *.out)
    """

    def __init__(self, filename: str):
        """
        Args:
            filename (str): Filename to parse
        """

        self.filename = filename
        self.data: Dict[str, Any] = {}
        self.data["errors"] = []
        self.data["warnings"] = {}

        self.text = ""
        with zopen(filename, mode="rt", encoding="ISO-8859-1") as f:
            self.text = f.read()

        # Parse the ORCA version
        version_match = read_pattern(
            self.text, {"key": r"^\s+Program Version ([0-9\.]+)"}, terminate_on_match=True
        ).get("key")
        if version_match is not None:
            self.data["version"] = version_match[0][0]
        else:
            self.data["version"] = "unknown"

        # Check if calculation finished
        completed_match = read_pattern(
            self.text,
            {"key": r"\s+\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*"},
            terminate_on_match=True,
        ).get("key")
        if completed_match is None:
            self.data["completed"] = False
        else:
            self.data["completed"] = True

        # Check runtime
        if self.data["completed"]:
            self._parse_runtime()
        else:
            self.data["runtime"] = None

        # Parse basic calculation parameters
        self._parse_calculation_parameters()

        # TODO: parse initial structure
        # Parse initial structure, including charge, spin, and number of electrons
        self._parse_initial_structure()

        # Parse the SCF
        self._parse_SCF()

        # Parse the partial charges (e.g. Mulliken) and dipoles
        self._parse_charges_and_dipoles()

        # Parse bond orders (from e.g. Mayer analysis)
        self._parse_bond_orders()

        # Check for various warnings
        self._parse_general_warnings()

        # Check for common errors
        self._parse_common_errors()

        # Check to see if PCM or SMD are present
        self._parse_solvent_info()

        # Parse the final energy
        self._parse_final_energy()

        # TODO: SHOULD check if job-type-specific parsing is actually necessary

        # Parse geometry optimization information
        self._parse_geometry_optimization()

        # Parse constrained optimization
        # TODO: SHOULD this be separate from normal geometry optimization stuff or no?
        self._parse_constrained_optimization()

        # Parse vibrational frequency analysis
        self._parse_frequency_analysis()

        # Parse gradient information
        self._parse_gradients()

        # Parse NBO information, if present
        self._parse_nbo()

    def _parse_initial_structure(self):
        # Tricky thing here: structure can be input from file, Cartesian, or internal coordinates
        # In inputs.py, assuming that we're only ever inputting Cartesian coordinates
        # That's probably lazy and not correct
        pass

    def _parse_calculation_parameters(self):
        if "parameters" not in self.data:
            self.data["parameters"] = dict()

        # Parse basis
        basis_matches = read_pattern(
            self.text,
            {
                "basis": r"Your calculation utilizes the basis: ([A-Za-z0-9\-\*\+\(\)]+)",
                "aux_basis": r"Your calculation utilizes the auxiliary basis: ([A-Za-z0-9/\-\*\+\(\)]+)"
            }
        )

        if basis_matches.get("basis") is None:
            self.data["parameters"]["basis"] = None
        else:
            self.data["parameters"]["basis"] = basis_matches["basis"][0][0]

        if basis_matches.get("aux_basis") is None:
            self.data["parameters"]["aux_basis"] = None
        else:
            self.data["parameters"]["aux_basis"] = basis_matches["aux_basis"][0][0]

        matches = read_pattern(
            self.text,
            {
                "charge": r"Total Charge\s+Charge\s+\.\.\.\.\s+(\d+)",
                "spin": r"Multiplicity\s+Mult\s+\.\.\.\.\s+(\d+)",
                "nelec": r"Number of Electrons\s+NEL\s+\.\.\.\.\s+(\d+)"
            }
        )

        # TODO: should we continue parsing if we can't get charge or spin?
        # We could also get this from the input information
        if matches.get("charge") is None:
            self.data["charge"] = None
        else:
            self.data["charge"] = int(matches["charge"][0][0])
        
        if matches.get("spin") is None:
            self.data["spin_multiplicity"] = None
            self.data["open_shell"] = None
        else:
            self.data["spin_multiplicity"] = int(matches["spin"][0][0])

            # Assess if the calculation is closed-shell or open-shell
            if self.data["spin_multiplicity"] != 1:
                self.data["open_shell"] = True
            else:
                self.data["open_shell"] = False
        
        if matches.get("nelec") is None:
            self.data["nelectrons"] = None
        else:
            self.data["nelectrons"] = int(matches["nelec"][0][0])

    def _parse_runtime(self):
        run_match = read_pattern(
            self.text,
            {"key": r"TOTAL RUN TIME: (\d+) days (\d+) hours (\d+) seconds (\d+) msec"},
            terminate_on_match=True,
        ).get("key")

        if run_match is None:
            self.data["runtime"] = None
        else:
            days = int(run_match[0][0])
            hours = int(run_match[0][1])
            minutes = int(run_match[0][2])
            seconds = int(run_match[0][3])
            milliseconds = int(run_match[0][4])

            total_seconds = 86400 * days + 3600 * hours + 60 * minutes + seconds + milliseconds / 1000
            self.data["runtime"] = total_seconds

    def _parse_SCF(self):
        header_pattern = r"\-+\s*SCF ITERATIONS\s*\-+"
        table_pattern = (r"\s*(?:(?:ITER\s+Energy\s+Delta\-E\s+Max\-DP\s+RMS\-DP\s+\[F,P\]\s+Damp)|"
                         r"(?:\s*\*\*\*[A-Za-z\s\-/]+\*\*\*)|"
                         r"(?:ITER\s+Energy\s+Delta\-E\s+Grad\s+Rot\s+Max\-DP\s+RMS\-DP)|"
                         r"(\s*\d+\s+[\-\.0-9]+\s+[\-\.0-9]+\s+[\-\.0-9]+\s+[\-\.0-9]+"
                         r"\s+[\-\.0-9]+\s+[\-\.0-9]+))\s*")
        footer_pattern = r"\s+\*+\s+\*\s+SUCCESS\s+\*\s+\*\s+SCF CONVERGED AFTER\s+\d+\s+CYCLES\s+\*\s+\*+"
        scf_match = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)

        scf = list()
        for one_scf in scf_match:
            this_scf = list()
            for point in one_scf:
                try:
                    parsed = point[0].strip().split()
                    if parsed[0] is None or parsed[0] == "None":
                        continue
                    this_scf.append(float(parsed[1]))
                except IndexError:
                    continue
            scf.append(this_scf)
        
        self.data["SCF"] = scf

    def _parse_orbitals(self):
        type_matches = read_pattern(
            self.text,
            {
                "open": r"\-+\s+ORBITAL ENERGIES\s+\-+\s+SPIN UP ORBITALS"
            }
        )

        header_pattern = r"\-+\s+ORBITAL ENERGIES\s+\-+\s+NO\s+OCC\s+E\(Eh\)\s+E\(eV\)"
        up_header = r"\s+SPIN UP ORBITALS\s+NO\s+OCC\s+E\(Eh\)\s+E\(eV\)"
        down_header = r"\s+SPIN DOWN ORBITALS\s+NO\s+OCC\s+E\(Eh\)\s+E\(eV\)"
        table_pattern = r"\s*\d+\s+([0-9\.]+)\s+([0-9\-\.]+)\s+([0-9\-\.]+)\s*\n"
        footer_pattern = r""

        # Molecule is open-shell
        if type_matches.get("open") is not None:
            if not self.data["open_shell"]:
                self.data["open_shell"] = True

            up_orbitals = list()
            down_orbitals = list()
            up_matches = read_table_pattern(
                self.text,
                up_header,
                table_pattern,
                footer_pattern
            )
            for orb_match in up_matches:
                this_orbitals = list()
                for orb in orb_match:
                    this_orbitals.append(
                        {
                            "occupancy": float(orb[0]),
                            "energy_Ha": float(orb[1])
                        }
                    )
                up_orbitals.append(this_orbitals)

            down_matches = read_table_pattern(
                self.text,
                down_header,
                table_pattern,
                footer_pattern
            )
            for orb_match in down_matches:
                this_orbitals = list()
                for orb in orb_match:
                    this_orbitals.append(
                        {
                            "occupancy": float(orb[0]),
                            "energy_Ha": float(orb[1])
                        }
                    )
                down_orbitals.append(this_orbitals)

            if len(up_orbitals) != len(down_orbitals):
                if "warnings" not in self.data:
                    self.data["warnings"] = list()
                self.data["warnings"].append("up_down_orbitals_dont_match")
            self.data["orbitals"] = list(zip(up_orbitals, down_orbitals))
        # Closed-shell
        else:
            orbital_matches = read_table_pattern(
                self.text,
                header_pattern,
                table_pattern,
                footer_pattern
            )

            orbitals = list()
            for orb_match in orbital_matches:
                this_orbitals = list()
                for orb in orb_match:
                    this_orbitals.append(
                        {
                            "occupancy": float(orb[0]),
                            "energy_Ha": float(orb[1])
                        }
                    )
                orbitals.append(this_orbitals)
            
            self.data["orbitals"] = orbitals

    def _parse_charges_and_dipoles(self):
        # Mulliken population analysis
        mulliken_open_match = read_pattern(
            self.text,
            {
                "open": r"\-+\s*MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\s*\-+"
            }
        )

        mulliken = list()
        # Molecule is open-shell
        if mulliken_open_match.get("open") is not None:
            if not self.data["open_shell"]:
                self.data["open_shell"] = True

            header_pattern = r"\-+\s*MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\s*\-+"
            table_pattern = r"\s*\d+\s+[A-Za-z]+\s*:\s+([0-9\-\.]+)\s+([0-9\-\.]+)\s*\n"
            footer_pattern = r"Sum of atomic charges\s+:\s+[0-9\.\-]+"

            mull_table_match = read_table_pattern(
                self.text,
                header_pattern,
                table_pattern,
                footer_pattern
            )

            for mull_match in mull_table_match:
                this_mulliken = list()
                for atom in mull_match:
                    this_mulliken.append((float(atom[0]), float(atom[1])))
                
                mulliken.append(this_mulliken)
        # Molecule is closed-shell
        else:
            header_pattern = r"\-+\s*MULLIKEN ATOMIC CHARGES\s*\-+"
            table_pattern = r"\s*\d+\s+[A-Za-z]+\s*:\s+([0-9\-\.]+)\s*\n"
            footer_pattern = r"Sum of atomic charges\s+:\s+[0-9\.\-]+"

            for mull_match in mull_table_match:
                this_mulliken = list()
                for atom in mull_match:
                    this_mulliken.append((float(atom[0])))
                
                mulliken.append(this_mulliken)
        
        self.data["mulliken"] = mulliken

        # Lowedin population analysis
        loewdin_open_match = read_pattern(
            self.text,
            {
                "open": r"\-+\s*LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS\s*\-+"
            }
        )

        loewdin = list()
        # Molecule is open-shell
        if loewdin_open_match.get("open") is not None:
            if not self.data["open_shell"]:
                self.data["open_shell"] = True

            header_pattern = r"\-+\s*LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS\s*\-+"
            table_pattern = r"\s*\d+\s+[A-Za-z]+\s*:\s+([0-9\-\.]+)\s+([0-9\-\.]+)\s*\n"
            footer_pattern = r""

            loew_table_match = read_table_pattern(
                self.text,
                header_pattern,
                table_pattern,
                footer_pattern
            )

            for loew_match in loew_table_match:
                this_loew = list()
                for atom in loew_match:
                    this_loew.append((float(atom[0]), float(atom[1])))
                
                loewdin.append(this_loew)
        # Molecule is closed-shell
        else:
            header_pattern = r"\-+\s*LOEWDIN ATOMIC CHARGES\s*\-+"
            table_pattern = r"\s*\d+\s+[A-Za-z]+\s*:\s+([0-9\-\.]+)\s*\n"
            footer_pattern = r"Sum of atomic charges\s+:\s+[0-9\.\-]+"

            for loew_match in loew_table_match:
                this_loew = list()
                for atom in loew_match:
                    this_loew.append((float(atom[0])))
                
                loewdin.append(this_loew)

        # Mayer population analysis



    def _parse_general_warnings(self):
        if "warnings" not in self.data:
            self.data["warnings"] = list()

        warning_matches = read_pattern(
            self.text,
            {
                "open_shell_RHF": r"WARNING: your system is open\-shell and RHF/RKS was chosen",
                "geom_opt": r"WARNING: Geometry Optimization",
                "S2_irrelevant": r"Warning: in a DFT calculation there is little theoretical justification to\s+"
                                 r"calculate <S\*\*2> as in Hartree-Fock theory.",
                "ulimit": r"The 'ulimit \-s' on the system is set to 'unlimited'. This may have negative performance\s+"
                          r"implications. Please set the stack size to the default value",
                "RFO_low_scaling": r"Warning: RFO finds a terribly low value for the scaling factor"
            }
        )

        if warning_matches.get("open_shell_RHF") is not None:
            self.data["warnings"].append("open_shell_RHF")
        if warning_matches.get("geom_opt") is not None:
            self.data["warnings"].append("geom_opt")
        if warning_matches.get("S2_irrelevant") is not None:
            self.data["warnings"].append("S2_irrelevant")
        if warning_matches.get("ulimit") is not None:
            self.data["warnings"].append("ulimit")
        if warning_matches.get("RFO_low_scaling") is not None:
            self.data["warnings"].append("RFO_low_scaling")

    def _parse_common_errors():
        # TODO
        pass

    def _parse_solvent_info(self):
        # TODO
        pass

    def _parse_final_energy(self):
        # TODO
        pass

    def _parse_geometry_optimization(self):
        # TODO
        pass

    def _parse_constrained_optimization(self):
        # TODO
        pass

    def _parse_frequency_analysis(self):
        # TODO
        pass

    def _parse_gradients(self):
        # TODO
        pass

    def _parse_bond_orders(self):
        # TODO
        pass

    def _parse_nbo(self):
        dfs = nbo_parser(self.filename)
        nbo_data = dict()
        for key, value in dfs.items():
            nbo_data[key] = [df.to_dict() for df in value]
        self.data["nbo_data"] = nbo_data


class ORCAPCMOutput(MSONable):
    pass


class ORCASMDOutput(MSONable):
    pass


class ORCAPropertyOutput(MSONable):
    pass


class ORCANBOOutput(MSONable):
    pass


class ORCATrajectoryOutput(MSONable):
    pass


class File47Output(MSONable):
    pass

