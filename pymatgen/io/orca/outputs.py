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

from pymatgen.core.structure import Molecule
from pymatgen.io.qchem.outputs import nbo_parser
from pymatgen.io.qchem.utils import (
    process_parsed_coords,
    process_parsed_fock_matrix,
    read_matrix_pattern,
    read_pattern,
    read_table_pattern,
)


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
        self.data["warnings"] = list()

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

        # Parse the SCF
        self._parse_SCF()

        # Parse the partial charges (e.g. Mulliken) and dipoles
        self._parse_charges_and_dipoles()

        # Parse bond orders (from e.g. Mayer analysis)
        self._parse_bond_orders()

        # Check for various warnings
        self._parse_general_warnings()

        # Check for common errors
        # TODO
        self._parse_common_errors()

        # Check to see if PCM or SMD are present
        self._parse_solvent_info()

        # Parse (Cartesian) structures
        self._parse_structures()

        # Parse gradient information
        self._parse_gradients()

        # Parse geometry optimization information
        if read_pattern(
            self.text,
            {"key": r"\*+\s+\*\s+Geometry Optimization Run\s+\*\s+\*+"}
        ).get("key") is not None:
            self._parse_geometry_optimization()

        if read_pattern(
            self.text,
            {"key": r"\-+\s+ORCA SCF HESSIAN\s+\-+"}
        ).get("key") is not None:
            # Parse vibrational frequency analysis
            self._parse_frequency_analysis()
            self._parse_thermo()

        # Parse NBO information, if present
        if read_pattern(
            self.text,
            {"key": r"\*+\s+NBO 7\.0\s+\*+\s+N A T U R A L\s+A T O M I C\s+O R B I T A L\s+A N D\s+"
                    r"N A T U R A L\s+B O N D\s+O R B I T A L\s+A N A L Y S I S\s+\*+"}
        ).get("key") is not None:
            self._parse_nbo()

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
                "charge": r"Total Charge\s+Charge\s+\.\.\.\.\s+([\-\d]+)",
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
                         r"(?:\s*\*\*\*\*\s*Energy Check signals convergence\s*\*\*\*\*)|"
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

        final_energy_match = read_pattern(
            self.text,
            {
                "final_energy": r"\-+\s+\-+\s*FINAL SINGLE POINT ENERGY\s+([0-9\-\.]+)\s*\-+\s+\-+",

            }
        )

        sp_energies = list()
        if final_energy_match.get("final_energy") is not None:
            for fe_match in final_energy_match["final_energy"]:
                sp_energies.append(float(fe_match[0]))
        self.data["sp_energies"] = sp_energies
        self.data["final_energy"] = sp_energies[-1]

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
            footer_pattern = r"Sum of atomic charges\s*:\s+[0-9\.\-]+"

            mull_table_match = read_table_pattern(
                self.text,
                header_pattern,
                table_pattern,
                footer_pattern
            )

            for mull_match in mull_table_match:
                this_mulliken = list()
                for atom in mull_match:
                    this_mulliken.append((float(atom[0])))
                
                mulliken.append(this_mulliken)
        
        self.data["mulliken_charges"] = mulliken

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

            loew_table_match = read_table_pattern(
                self.text,
                header_pattern,
                table_pattern,
                footer_pattern
            )

            for loew_match in loew_table_match:
                this_loew = list()
                for atom in loew_match:
                    this_loew.append((float(atom[0])))
                
                loewdin.append(this_loew)
        self.data["loewdin_charges"] = loewdin

        # Mayer population analysis
        header_pattern = r"ATOM\s+NA\s+ZA\s+QA\s+VA\s+BVA\s+FA\s*"
        table_pattern = (r"\s*\d+\s+[A-Za-z]+\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)"
                         r"\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s*\n")
        footer_pattern = r""

        mayer = list()
        mayer_match = read_table_pattern(
            self.text,
            header_pattern,
            table_pattern,
            footer_pattern
        )

        for match in mayer_match:
            this_mayer = list()
            for atom in match:
                this_mayer.append(
                    {
                        "gross_population": float(atom[0]),
                        "nuclear_charge": float(atom[1]),
                        "gross_atomic_charge": float(atom[2]),
                        "mayer_total_valence": float(atom[3]),
                        "mayer_bonded_valence": float(atom[4]),
                        "mayer_free_valence": float(atom[5])
                    }
                )
            mayer.append(this_mayer)
        self.data["mayer_charges"] = mayer

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

    def _parse_thermo(self):
        thermo_matches = read_pattern(
            self.text,
            {
                "temperature": r"Temperature\s+\.\.\.\s+([0-9\.]+)\s+K",
                "pressure": r"Pressure\s+\.\.\.\s+([0-9\.]+)\s+atm",
                "electronic_energy": r"Electronic energy\s+\.\.\.\s+([0-9\.\-]+)\s+Eh",
                "zpe": r"Zero point energy\s+\.\.\.\s+([0-9\.\-]+)\s+Eh\s+([0-9\.\-]+)\s+kcal/mol",
                "therm_vib_corr": (r"Thermal vibrational correction\s+\.\.\.\s+([0-9\.\-]+)\s+Eh\s+"
                                   r"([0-9\.\-]+)\s+kcal/mol"),
                "therm_rot_corr": (r"Thermal rotational correction\s+\.\.\.\s+([0-9\.\-]+)\s+Eh\s+"
                                   r"([0-9\.\-]+)\s+kcal/mol"),
                "therm_trans_corr": (r"Thermal translational correction\s+\.\.\.\s+([0-9\.\-]+)\s+Eh\s+"
                                     r"([0-9\.\-]+)\s+kcal/mol"),
                "tot_internal": r"Total thermal energy\s+([0-9\-\.]+)\s+Eh",
                "therm_total": r"Total thermal correction\s+([0-9\.\-]+)\s+Eh\s+([0-9\.\-]+)\s+kcal/mol",
                "correction_total": r"Total correction\s+([0-9\.\-]+)\s+Eh\s+([0-9\.\-]+)\s+kcal/mol",
                "thermal_enthalpy": r"Thermal Enthalpy correction\s+\.\.\.\s+([0-9\.\-]+)\s+Eh"
                                    r"\s+([0-9\.\-]+)\s+kcal/mol",
                "enthalpy_total": r"Total Enthalpy\s+\.\.\.\s+([0-9\.\-]+)\s+Eh",
                "entropy_elec": r"Electronic entropy\s+\.\.\.\s+([0-9\.\-]+)\s+Eh\s+([0-9\.\-]+)\s+kcal/mol",
                "entropy_vib": r"Vibrational entropy\s+\.\.\.\s+([0-9\.\-]+)\s+Eh\s+([0-9\.\-]+)\s+kcal/mol",
                "entropy_rot": r"Rotational entropy\s+\.\.\.\s+([0-9\.\-]+)\s+Eh\s+([0-9\.\-]+)\s+kcal/mol",
                "entropy_trans": r"Translational entropy\s+\.\.\.\s+([0-9\.\-]+)\s+Eh\s+([0-9\.\-]+)\s+kcal/mol",
                "entropy_total": r"Final entropy term\s+\.\.\.\s+([0-9\.\-]+)\s+Eh\s+([0-9\.\-]+)\s+kcal/mol",
                "gibbs_free_energy": r"Final Gibbs free energy\s+\.\.\.\s+([0-9\.\-]+)\s+Eh",
                "gibbs_energy_diff": r"G\-E\(el\)\s+\.\.\.\s+([0-9\.\-]+)\s+Eh\s+([0-9\.\-]+)\s+kcal/mol"
            }
        )

        if thermo_matches.get("temperature") is not None:
            self.data["thermo_temperature"] = [float(i[0]) for i in thermo_matches.get("temperature")]
        if thermo_matches.get("pressure") is not None:
            self.data["thermo_pressure"] = [float(i[0]) for i in thermo_matches.get("pressure")]
        if thermo_matches.get("electronic_energy") is not None:
            self.data["electronic_energy"] = [float(i[0]) for i in thermo_matches.get("electronic_energy")]
        if thermo_matches.get("zpe") is not None:
            self.data["zero_point_energy"] = [float(i[0]) for i in thermo_matches.get("zpe")]
        if thermo_matches.get("thermo_vib_corr") is not None:
            self.data["thermal_vibration_correction"] = [float(i[0]) for i in thermo_matches.get("thermo_vib_corr")]
        if thermo_matches.get("thermo_rot_corr") is not None:
            self.data["thermal_rotation_correction"] = [float(i[0]) for i in thermo_matches.get("thermo_rot_corr")]
        if thermo_matches.get("thermo_trans_corr") is not None:
            self.data["thermal_translation_correction"] = [float(i[0]) for i in thermo_matches.get("thermo_trans_corr")]
        if thermo_matches.get("tot_internal") is not None:
            self.data["total_internal_energy"] = [float(i[0]) for i in thermo_matches.get("tot_internal")]
        if thermo_matches.get("therm_total") is not None:
            self.data["total_thermal_energy"] = [float(i[0]) for i in thermo_matches.get("therm_total")]
        if thermo_matches.get("correction_total") is not None:
            self.data["total_thermo_correction"] = [float(i[0]) for i in thermo_matches.get("correction_total")]
        if thermo_matches.get("thermal_enthalpy") is not None:
            self.data["thermal_enthalpy"] = [float(i[0]) for i in thermo_matches.get("thermal_enthalpy")]
        if thermo_matches.get("enthalpy_total") is not None:
            self.data["total_enthalpy"] = [float(i[0]) for i in thermo_matches.get("enthalpy_total")]
        if thermo_matches.get("entropy_elec") is not None:
            self.data["electronic_entropy"] = [float(i[0]) for i in thermo_matches.get("entropy_elec")]
        if thermo_matches.get("entropy_vib") is not None:
            self.data["vibrational_entropy"] = [float(i[0]) for i in thermo_matches.get("entropy_vib")]
        if thermo_matches.get("entropy_rot") is not None:
            self.data["rotational_entropy"] = [float(i[0]) for i in thermo_matches.get("entropy_rot")]
        if thermo_matches.get("entropy_trans") is not None:
            self.data["translational_entropy"] = [float(i[0]) for i in thermo_matches.get("entropy_trans")]
        if thermo_matches.get("entropy_total") is not None:
            self.data["total_entropy"] = [float(i[0]) for i in thermo_matches.get("entropy_total")]
        if thermo_matches.get("gibbs_free_energy") is not None:
            self.data["gibbs_free_energy"] = [float(i[0]) for i in thermo_matches.get("gibbs_free_energy")]
        if thermo_matches.get("gibbs_energy_diff") is not None:
            self.data["g_minus_e"] = [float(i[0]) for i in thermo_matches.get("gibbs_energy_diff")]        

        symmetry_match = read_pattern(
            self.text,
            {
                "pg_sn": r"Point Group:\s+([A-Za-z\d]+),\s+Symmetry Number:\s+(\d+)",
                "rot_consts": r"Rotational constants in cm-1:\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)"
            }
        )

        if symmetry_match.get("pg_sn") is not None:
            self.data["point_group"] = symmetry_match["pg_sn"][0][0]
            self.data["symmetry_number"] = int(symmetry_match["pg_sn"][0][1])

        header_pattern = r"\s*\-+\s*"
        row_pattern = r"\s*\|\s+sn=\s+(\d+)\s+\|\s+S\(rot\)=\s+([0-9\-\.]+)\s+Eh\s+([0-9\-\.]+)\s+kcal/mol\|\s*"
        footer_pattern = r"\s*\-+"
        entropy_by_sn = read_table_pattern(
            self.text,
            header_pattern,
            row_pattern,
            footer_pattern
        )
        entropy_by_sn = list()
        for one_s_sn in entropy_by_sn:
            this_s_by_sn = dict()
            for row in one_s_sn:
                sn = int(row[0])
                s = float(row[1])
                this_s_by_sn[sn] = s
            entropy_by_sn.append(this_s_by_sn)
        
        self.data["rotational_entropy_by_symmetry_number"] = entropy_by_sn

    def _parse_geometry_optimization(self):
        # First, parse basic optimization settings
        opt_settings_match = read_pattern(
            self.text,
            {
                "update_method": r"Update method\s+Update\s+\.\.\.\.\s+([^\n]+)\n",
                "coordinates": r"Choice of coordinates\s+CoordSys\s+\.\.\.\.\s+([^\n]+)\n",
                "initial_hessian": r"Initial Hessian\s+InHess\s+\.\.\.\.\s+([^\n]+)\n",
                "energy_tolerance": r"Energy Change\s+TolE\s+\.\.\.\.\s+([0-9\.\-e]+)\s+Eh",
                "max_gradient": r"Max\. Gradient\s+TolMAXG\s+\.\.\.\.\s+([0-9\.\-e]+)\s+Eh\bohr",
                "rms_gradient": r"RMS Gradient\s+TolRMSG\s+\.\.\.\.\s+([0-9\.\-e]+)\s+Eh/bohr",
                "max_displacement": r"Max\. Displacement\s+TolMAXD\s+\.\.\.\.\s+([0-9\.\-e]+)\s+bohr",
                "rms_displacement": r"RMS Displacement\s+TolRMSD\s+\.\.\.\.\s+([0-9\.\-e]+)\s+bohr",
                "strict_convergence": r"Strinct Convergence\s+\.\.\.\.\s+(False|True)"
            }
        )

        # Basic parameters
        if opt_settings_match.get("update_method") is not None:
            self.data["geom_opt_update_method"] = opt_settings_match["update_method"][0][0].strip()
        if opt_settings_match.get("coordinates") is not None:
            self.data["geom_opt_coordinate_system"] = opt_settings_match["coordinates"][0][0].strip()
        if opt_settings_match.get("initial_hessian") is not None:
            self.data["geom_opt_initial_hessian"] = opt_settings_match["initial_hessian"][0][0].strip()

        # Convergence criteria
        if opt_settings_match.get("energy_tolerance") is not None:
            self.data["geom_opt_energy_tolerance"] = float(opt_settings_match["energy_tolerance"][0][0])
        if opt_settings_match.get("max_gradient") is not None:
            self.data["geom_opt_max_gradient"] = float(opt_settings_match["max_gradient"][0][0])
        if opt_settings_match.get("rms_gradient") is not None:
            self.data["geom_opt_rms_gradient"] = float(opt_settings_match["rms_gradient"][0][0])
        if opt_settings_match.get("max_displacement") is not None:
            self.data["geom_opt_max_displacement"] = float(opt_settings_match["max_displacement"][0][0])
        if opt_settings_match.get("rms_displacement") is not None:
            self.data["geom_opt_rms_displacement"] = float(opt_settings_match["rms_displacement"][0][0])
        if opt_settings_match.get("strict_convergence") is not None:
            match = opt_settings_match["strict_convergence"][0][0]
            if match == "True":
                self.data["geom_opt_strict_convergence"] = True
            else:
                self.data["geom_opt_strict_convergence"] = False

    def _parse_frequency_analysis(self):
        # Frequencies
        header_pattern = (r"\-+\s+VIBRATIONAL FREQUENCIES\s+\-+\s+Scaling factor for frequencies\s+"
                          r"=\s+[\-\.0-9]+\s+\(already applied\!\)\s*")
        row_pattern = r"\s*\d+:\s+([0-9\.\-]+)\s+cm\*\*\-1(?: \*\*\*imaginary mode\*\*\*)?\s*"
        footer_pattern = r""

        freq_matches = read_table_pattern(
            self.text,
            header_pattern,
            row_pattern,
            footer_pattern
        )
        if len(freq_matches) == 0:
            self.data["frequencies"] = None
        else:
            # Should only be one IR spectrum
            # In case that's not true, take the last one
            freqs = list()
            for row in freq_matches[-1]:
                freqs.append(float(row[0]))
            self.data["frequencies"] = freqs

        # Normal modes
        normal_mode_table_match = read_pattern(
            self.text,
            {
                "key": (
                    r"\-+\s+NORMAL MODES\s+\-+\s+These modes are the Cartesian displacements weighted by the diagonal "
                    r"matrix\s+M\(i,i\)=1/sqrt\(m\[i\]\) where m\[i\] is the mass of the displaced atom\s+"
                    r"Thus, these vectors are normalized but \*not\* orthogonal\s+"
                    r"((?:(?:\s*(?:\d+\s*){1,6}\n)|(?:\s*\d+\s+([\-0-9]+\.[0-9]+\s+){1,6}\s*))+)"
                )
            }
        )
        if normal_mode_table_match.get("key") is not None:
            for nmt in normal_mode_table_match["key"]:
                table = nmt[0]
                # Separate by chunk of normal modes
                mode_chunk_match = read_pattern(
                    table,
                    {
                        "key": r"((?:\s*\d+\s+(?:[\-0-9]+\.[0-9]+\s+){1,6}\s*)+)"
                    }
                )
                # Not doing the normal 'if x.get("key") is not None'
                # If we get this far, the chunk SHOULD be there
                modes = list()
                for chunk in mode_chunk_match["key"]:
                    lines = chunk[0].split("\n")
                    lines_contents = [line.strip().split() for line in lines]
                    num_modes = max([len(x) for x in lines_contents]) - 1
                    these_modes = [[] for i in range(num_modes)]
                    for contents in lines_contents:
                        # Edge case - end line of section
                        if len(contents) < 2:
                            continue
                        
                        for ii, c in enumerate(contents[1:]):
                            these_modes[ii].append(float(c))
                    for mode in these_modes:
                        as_array = np.array(mode).reshape((-1, 3))
                        modes.append(as_array.tolist())
                self.data["normal_modes"] = modes

        # IR spectra
        header_pattern = (
            r"\-+\s+IR SPECTRUM\s+\-+\s+Mode\s+freq\s+eps\s+Int\s+T\*\*2\s+TX\s+TY\s+TZ\s+"
            r"cm\*\*-1\s+L/\(mol\*cm\)\s+km/mol\s+a\.u\.\s+\-+\s*"
        )
        row_pattern = (
            r"\s*(\d+):\s+([0-9\-i\.]+)\s+([\-\.0-9]+)\s+([\-\.0-9]+)\s+([\-\.0-9]+)\s+"
            r"\(\s*([\-\.0-9]+)\s+([\-\.0-9]+)\s+([\-\.0-9]+)\s*\)\s*"
        )
        footer_pattern = r""

        ir_matches = read_table_pattern(
            self.text,
            header_pattern,
            row_pattern,
            footer_pattern
        )

        if len(ir_matches) == 0:
            self.data["ir_spectrum"] = None
        else:
            # Should only be one IR spectrum
            # In case that's not true, take the last one
            frequencies = list()
            for row in ir_matches[-1]:
                if "i" in row[1].lower():
                    freq = -1 * float(row[1].lower().replace("i", ""))
                else:
                    freq = float(row[1])
                frequency = {
                    "frequency": freq,
                    "epsilon": float(row[1]),
                    "intensity": float(row[2]),
                    "T**2": float(row[3]),
                    "Tx": float(row[4]),
                    "Ty": float(row[5]),
                    "Tz": float(row[6])
                }
                frequencies.append(frequency)
            self.data["ir_spectrum"] = frequencies

        # Vibrational energy contributions
        if self.data["ir_spectrum"] is not None:
            header_pattern = r""
            row_pattern = r"freq\.\s+([0-9\.\-i]+)\s+E\(vib\)\s+\.\.\.\s+([0-9\.\-]+)"
            footer_pattern = r""

            vib_energy_match = read_table_pattern(
                self.text,
                header_pattern,
                row_pattern,
                footer_pattern
            )

            # Likewise, there should only be one section with vibrational energy contributions
            if len(vib_energy_match) > 0:
                vib_contributions = list()
                for row in vib_energy_match[-1]:
                    if "i" in row[0].lower():
                        freq = -1 * float(row[0].lower().replace("i", ""))
                    else:
                        freq = float(row[0])
                    
                    vib_energy = float(row[1])

                    vib_contributions.append((freq, vib_energy))
                self.data["vibrational_contributions"] = vib_contributions

    def _parse_bond_orders(self):
        def _get_bond_order_from_symbol(strength: str):
            if strength == "S":
                return 1.0
            elif strength == "D":
                return 2.0
            elif strength == "T":
                return 3.0
            elif strength == "Q":
                return 4.0
            elif strength == "5":
                return 5.0
            else:
                return 6.0
            
        # Mayer bond order
        mayer_match = read_pattern(
            self.text,
            {
                "key": r"\s+Mayer bond orders larger than [0-9\.\-]+\s+"
                       r"((?:B\(\s+[0-9]+\-[A-Za-z]+\s+,\s+[0-9]+\-[A-Za-z]+\s+\)\s+:\s+[0-9\.\-]+\s+)+)"
            }
        )
        if mayer_match.get("key") is not None:
            mayer_bonds = list()
            for mm in mayer_match["key"]:
                indiv_bond_match = read_pattern(
                    mm[0],
                    {
                        "bond": r"B\(\s+([0-9]+)\-([A-Za-z]+)\s+,\s+([0-9]+)\-([A-Za-z]+)\s+\)\s+:\s+([0-9\.\-]+)\s+"
                    }
                )
                for bond in indiv_bond_match.get("bond"):
                    mayer_bonds.append((int(bond[0]), int(bond[2]), float(bond[4])))
            self.data["mayer_bonds"] = mayer_bonds

        # NBO bonding summary
        nbo_bond_match = read_pattern(
            self.text,
            {
                "key": r"\$CHOOSE((?:.|\s)+?)\$END",
            }
        )

        if nbo_bond_match.get("key") is not None:
            contents = nbo_bond_match["key"][0][0]

            choose_sec_match = read_pattern(
                contents,
                {
                    "lone": r"\s*LONE\s+((?:\d+\s+)+)\s*END",
                    "bond": r"\s*BOND\s+((?:[STDQ56] \d+ \d+\s+)+)\s*END",
                    "3c": r"\s*3C\s+((?:[STDQ56] \d+ \d+ \d+\s+)+)\s*END"
                }
            )

            if choose_sec_match.get("lone") is not None:
                lps = choose_sec_match["lone"][0][0]
                nbo_lone_pairs = dict()
                for i in lps.strip().split():
                    if int(i) not in nbo_lone_pairs:
                        nbo_lone_pairs[int(i)] = 1
                    else:
                        nbo_lone_pairs[int(i)] += 1
                self.data["nbo_lone_pairs"] = nbo_lone_pairs

            if choose_sec_match.get("bond") is not None:
                bds = choose_sec_match["bond"][0][0]
                nbo_bonds = list()
                contents = bds.strip().split()
                num_bonds = len(contents) / 3
                for i in range(int(num_bonds)):
                    strength = contents[i * 3]
                    order = _get_bond_order_from_symbol(strength)
                    atom1 = int(contents[i * 3 + 1])
                    atom2 = int(contents[i * 3 + 2])
                    nbo_bonds.append((atom1, atom2, order))
                self.data["nbo_bonds"] = nbo_bonds

            if choose_sec_match.get("3c") is not None:
                threec = choose_sec_match["3c"][0][0]
                nbo_three_center = list()
                contents = threec.strip().split()
                num_3c = len(contents) / 4
                for i in range(int(num_3c)):
                    strength = contents[i * 4]
                    order = _get_bond_order_from_symbol(strength)
                    atom1 = int(contents[i * 4 + 1])
                    atom2 = int(contents[i * 4 + 2])
                    atom3 = int(contents[i * 4 + 3])
                    nbo_three_center.append((atom1, atom2, atom3, order))
                self.data["nbo_three_center"] = nbo_three_center

    def _parse_nbo(self):
        dfs = nbo_parser(self.filename)
        nbo_data = dict()
        for key, value in dfs.items():
            nbo_data[key] = [df.to_dict() for df in value]
        self.data["nbo_data"] = nbo_data

    def _parse_structures(self):
        header_pattern = r"\-+\s+CARTESIAN COORDINATES \(ANGSTROEM\)\s+\-+"
        row_pattern = r"\s*([A-Za-z]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s*"
        footer_pattern = r""

        structure_match = read_table_pattern(
            self.text,
            header_pattern,
            row_pattern,
            footer_pattern
        )

        if self.data["charge"] is None:
            charge = 0.0
        else:
            charge = self.data["charge"]
        
        spin = self.data.get("spin_multiplicity")

        geometries = list()
        molecules = list()
        for structure in structure_match:
            this_species = list()
            this_coords = list()
            for row in structure:
                this_species.append(row[0])
                this_coords.append(
                    [
                        float(row[1]),
                        float(row[2]),
                        float(row[3])
                    ]
                )
            geometries.append(this_coords)
            molecules.append(
                Molecule(
                    this_species,
                    this_coords,
                    charge=charge,
                    spin_multiplicity=spin
                )
            )
        
        self.data["geometries"] = geometries
        self.data["molecules"] = molecules
        self.data["initial_molecule"] = molecules[0]
        self.data["molecule_from_final_geometry"] = molecules[-1]

    def _parse_gradients(self):
        header_pattern = r"\-+\s+CARTESIAN GRADIENT\s+\-+\s+"
        row_pattern = r"\s*\d+\s+([A-Za-z]+)\s+:\s+([0-9\-\.]+)\s+([0-9\-\.]+)\s+([0-9\-\.]+)\s*"
        footer_pattern = r""

        grad_matches = read_table_pattern(
            self.text,
            header_pattern,
            row_pattern,
            footer_pattern
        )

        gradients = list()
        for grad_match in grad_matches:
            this_gradient = list()
            for row in grad_match:
                this_gradient.append(
                    [
                        float(row[1]),
                        float(row[2]),
                        float(row[3])
                    ]
                )
            gradients.append(this_gradient)
        self.data["gradients"] = gradients

    def _parse_solvent_info(self):
        cpcm_matches = read_pattern(
            self.text,
            {
                "cpcm": r"CPCM parameters:\s+Epsilon\s+\.\.\.\s+([0-9\.]+)\s+Refrac\s+\.\.\.\s+([0-9\.]+)\s+"
                        r"Rsolv\s+\.\.\.\s+([0-9\.]+)\s+Surface type\s+\.\.\.\s+([A-Za-z ]+)\s+"
                        r"Epsilon function type\s+\.\.\.\s+([A-Za-z ]+)",
                "solvent": r"Solvent:\s+\.\.\.\s+([A-Za-z0-9\-\(\),/ ]+)",
                "smd_cds": r"SMD\-CDS solvent descriptors:\s+Soln\s+\.\.\.\s+([0-9\.]+)\s+Soln25\s+\.\.\.\s+([0-9\.]+)"
                           r"\s+Sola\s+\.\.\.\s+([0-9\.]+)\s+Solb\s+\.\.\.\s+([0-9\.]+)\s+Solg\s+\.\.\.\s+([0-9\.]+)"
                           r"\s+Solc\s+\.\.\.\s+([0-9\.]+)\s+Solh\s+\.\.\.\s+([0-9\.]+)"
            }
        )

        if cpcm_matches.get("cpcm") is not None:
            cpcm = cpcm_matches["cpcm"][0]
            self.data["cpcm_epsilon"] = float(cpcm[0])
            self.data["cpcm_n"] = float(cpcm[1])
            self.data["cpcm_rsolv"] = float(cpcm[2])
            self.data["cpcm_surface_type"] = cpcm[3].strip()
            self.data["cpcm_epsilon_function_type"] = cpcm[4].strip()
        if cpcm_matches.get("solvent") is not None:
            self.data["smd_solvent"] = cpcm_matches["solvent"][0][0].strip()
        if cpcm_matches.get("smd_cds") is not None:
            smd_cds = cpcm_matches["smd_cds"][0]
            self.data["smd_n"] = float(smd_cds[0])
            self.data["smd_n25"] = float(smd_cds[1])
            self.data["smd_a"] = float(smd_cds[2])
            self.data["smd_b"] = float(smd_cds[3])
            self.data["smd_g"] = float(smd_cds[4])
            self.data["smd_c"] = float(smd_cds[5])
            self.data["smd_h"] = float(smd_cds[6])

    def _parse_common_errors(self):
        # TODO
        pass


class ORCAPCMOutput(MSONable):
    """
    Class to parse ORCA PCM output files (typically *.cpcm)
    """

    def __init__(self, filename: str, parse_surface_points: bool = False):
        """
        Args:
            filename (str): Filename to parse
        """

        self.filename = filename
        self.data: Dict[str, Any] = {}
        self.parse_surface_points = parse_surface_points

        self.text = ""
        with zopen(filename, mode="rt", encoding="ISO-8859-1") as f:
            self.text = f.read()

        self._parse_basic_information()
        
        self._parse_radii()

        if self.parse_surface_points:
            self._parse_surface_points()

    def _parse_basic_information(self):
        basic_matches = read_pattern(
            self.text,
            {
                "num_atoms": r"(\d+)\s+#\s+Number of atoms",
                "num_surf_points": r"(\d+)\s+#\s+Number of surface points",
                "surf_type": r"(\d+)\s+#\s+Surface type",
                "epsilon_func_type": r"(\d+)\s+#\s+Epsilon function type",
                "num_leb_points": r"(\d+)\s+#\s+Number of Leb\. points",
                "volume": r"\s*([0-9\.]+)\s+#\s+Volume",
                "area": r"\s*([0-9\.]+)\s+#\s+Area",
                "cpcm_dielec_energy": r"\s*([0-9\.\-]+)\s+#\s+CPCM dielectric energy",
                "1e_operator_energy": r"\s*([0-9\.\-]+)\s+#\s+One\-electron operator energy"
            }
        )

        if basic_matches.get("num_atoms") is not None:
            self.data["num_atoms"] = int(basic_matches["num_atoms"][0][0])
        if basic_matches.get("num_surf_point") is not None:
            self.data["num_surf_points"] = int(basic_matches["num_surf_points"][0][0])
        if basic_matches.get("surf_type") is not None:
            self.data["surf_type"] = int(basic_matches["surf_type"][0][0])
        if basic_matches.get("epsilon_func_type") is not None:
            self.data["epsilon_func_type"] = int(basic_matches["epsilon_func_type"][0][0])
        if basic_matches.get("num_leb_points") is not None:
            self.data["num_leb_points"] = int(basic_matches["num_leb_points"][0][0])
        if basic_matches.get("volume") is not None:
            self.data["volume"] = float(basic_matches["volume"][0][0])
        if basic_matches.get("area") is not None:
            self.data["area"] = float(basic_matches["area"][0][0])
        if basic_matches.get("cpcm_dielec_energy") is not None:
            self.data["cpcm_dielec_energy"] = float(basic_matches["cpcm_dielec_energy"][0][0])
        if basic_matches.get("1e_operator_energy") is not None:
            self.data["1e_operator_energy"] = float(basic_matches["1e_operator_energy"][0][0])

    def _parse_radii(self):
        header_pattern = r"#\-+\s+#\s+CARTESIAN COORDINATES\s+\(A\.U\.\)\s+\+\s+RADII\s+\(A\.U\.\)\s+#\-+\s*"
        row_pattern = r"\s*(?:[0-9\-]+\.[0-9]+)\s+(?:[0-9\-]+\.[0-9]+)\s+(?:[0-9\-]+\.[0-9]+)\s+([0-9\.\-]+)"
        footer_pattern = r""

        cart_coords_match = read_table_pattern(
            self.text,
            header_pattern,
            row_pattern,
            footer_pattern
        )

        for table in cart_coords_match:
            radii = list()
            for row in table:
                radii.append(float(row[0]))
            self.data["pcm_radii"] = radii
            # There should only be one of these tables, so stop after parsing the first one
            break

    def _parse_surface_points(self):
        header_pattern = (
            r"#\-+\s+#\s+SURFACE POINTS \(A\.U\.\)\s+\(Hint \- charge NOT scaled by FEps\)\s+#\-+\s+"
            r"X\s+Y\s+Z\s+area\s+potential\s+charge\s+w_leb\s+Switch_F\s+G_width\s+atom\s*"
        )
        row_pattern = (
            r"([\-\.0-9eE]+)\s+([\-\.0-9eE]+)\s+([\-\.0-9eE]+)\s+([\-\.0-9eE]+)\s+([\-\.0-9eE]+)"
            r"\s+([\-\.0-9eE]+)\s+([\-\.0-9eE]+)\s+([\-\.0-9eE]+)\s+([\-\.0-9eE]+)\s+(\d+)\s*"
        )
        footer_pattern = r""

        surf_point_matches = read_table_pattern(
            self.text,
            header_pattern,
            row_pattern,
            footer_pattern
        )

        for table in surf_point_matches:
            points = list()
            for row in table:
                points.append(
                    {
                        "x": float(row[0]),
                        "y": float(row[1]),
                        "z": float(row[2]),
                        "area": float(row[3]),
                        "potential": float(row[4]),
                        "charge": float(row[5]),
                        "w_leb": float(row[6]),
                        "switch_f": float(row[7]),
                        "g_width": float(row[8]),
                        "atom": int(row[9])
                    }
                )

            self.data["surface_points"] = points
            # There should only be one of these tables, so stop after parsing the first one
            break


class ORCASMDOutput(MSONable):
    """
    Class to parse ORCA SMD output files (typically *.smd.out)
    """

    def __init__(self, filename: str):
        """
        Args:
            filename (str): Filename to parse
        """

        self.filename = filename
        self.data: Dict[str, Any] = {}

        self.text = ""
        with zopen(filename, mode="rt", encoding="ISO-8859-1") as f:
            self.text = f.read()

        self._parse_element_sasa_energy()
        self._parse_pair_cotsasa_energy()

        # Parse nonaqueous solvent contribution
        na_solv_contrib_match = read_pattern(
            self.text,
            {
                "key": (
                    r"NONAQUEOUS SOLVENT CONTRIBUTION \(kcal/mol\):\s+"
                    r"\(proportional to SASA, atomic number independent\)\s+\-+\s+Subtotal:\s+([0-9\.\-]+)\s+\-+"
                )
            }
        )
        if na_solv_contrib_match.get("key") is not None:
            self.data["nonaqueous_solvent_contribution"] = float(na_solv_contrib_match["key"][0][0])

        # Parse total CDS energy
        cds_total_match = read_pattern(
            self.text,
            {
                "key": r"\s*CDS TOTAL \(kcal/mol\):\s+([0-9\.\-]+)\s+\.+"
            }
        )
        if cds_total_match.get("key") is not None:
            self.data["cds_total_energy"] = float(cds_total_match[0][0])
        
    def _parse_element_sasa_energy(self):
        sasa_matches = read_pattern(
            self.text,
            {
                "key": (
                    r"\-+\s+Element\s+SASA\s+Sigma Z\s+Total\s+Z\s+Ang\*\*2\s+cal/Ang\*\*2/mol\s+kcal/mol\s+\-+\s+"
                    r"((?:\s*[A-za-z]+\s+[0-9\.\-]+\s+[0-9\.\-]+\s+[0-9\.\-]+)+)"
                    r"\s+Subtotal:\s+([\-\.0-9]+)\s+([0-9\.\-]+)\s+\-+"
                )
            }
        )

        if sasa_matches.get("key") is not None:
            # There should be only one match
            match = sasa_matches["key"][0]
            rows = match[0]
            row_matches = read_pattern(
                rows,
                {
                    "key": r"\s*([A-Za-z]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s*"
                }
            )
            elem_contribs = dict()
            for row in row_matches.get("key", list()):
                element = row[0]
                sasa = float(row[1])
                sigma_z = float(row[2])
                energy = float(row[3])
                elem_contribs[element] = {
                    "sasa": sasa,
                    "sigma_z": sigma_z,
                    "energy": energy
                }
            self.data["single_atom_contributions"] = elem_contribs

            self.data["sasa_total"] = float(match[1])

            self.data["total_single_atom_contribution"] = float(match[2])

    def _parse_pair_cotsasa_energy(self):
        two_atom_matches = read_pattern(
            self.text,
            {
                "key": (
                    r"\-+\s+Elements\s+COT\*SASA\s+Sigma Z,Z'\s+Total\s+Z,Z'\s+Ang\*\*2\s+cal/Ang\*\*2/mol\s+kcal/mol\s+\-+\s+"
                    r"((?:\s*[A-Za-z]+\s+[A-Za-z]+\s+[0-9\.\-]+\s+[0-9\.\-]+\s+[0-9\.\-]+)+)"
                    r"\s+Subtotal:\s+([0-9\.\-]+)\s+\-+"
                )
            }
        )

        if two_atom_matches.get("key") is not None:
            # There should be only one match
            match = two_atom_matches["key"][0]
            
            rows = match[0]
            rows_matches = read_pattern(
                rows,
                {
                    "key": r"\s*([A-Za-z]+)\s+([A-Za-z]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)"
                }
            )

            pair_contributions = dict()
            for row in rows_matches.get("key", list()):
                elem_1 = row[0]
                elem_2 = row[1]
                name = f"{elem_1}-{elem_2}"
                cotsasa = float(row[2])
                sigma_zz = float(row[3])
                energy = float(row[4])
                pair_contributions[name] = {
                    "cot_sasa": cotsasa,
                    "sigma_z_z'": sigma_zz,
                    "energy": energy
                }
            self.data["atom_pair_contributions"] = pair_contributions

            self.data["total_pair_contribution"] = float(match[1])


class ORCAPropertyOutput(MSONable):
    """
    Class to parse ORCA property output files (typically *_property.txt)
    """

    def __init__(self, filename: str):
        """
        Args:
            filename (str): Filename to parse
        """

        self.filename = filename
        self.data: Dict[str, Any] = {}

        self.text = ""
        with zopen(filename, mode="rt", encoding="ISO-8859-1") as f:
            self.text = f.read()
            self.sections = self.text.split("$")

        self._parse_calculation_info()
        self._parse_SCF_energy()
        self._parse_DFT_energy()
        self._parse_solvation_details()
        self._parse_mayer_pop()
        self._parse_SCF_electric_properties()
        self._parse_hessian()
        self._parse_thermochemistry()
        self._parse_geometries()
    
    def _parse_calculation_info(self):
        for section in self.sections:
            sec_match = read_pattern(
                section,
                {
                    "key": r"Calculation_Info"
                }
            )
            if sec_match.get("key") is not None:
                contents_match = read_pattern(
                    section,
                    {
                        "geom_index": r"geom\. index: (\d+)",
                        "multiplicity": r"Multiplicity:\s+(\d+)",
                        # ""  # TODO
                    }
                )

    def _parse_SCF_energy(self):
        pass

    def _parse_DFT_energy(self):
        pass

    def _parse_solvation_details(self):
        pass

    def _parse_mayer_pop(self):
        pass

    def _parse_SCF_electric_properties(self):
        pass

    def _parse_hessian(self):
        pass

    def _parse_thermochemistry(self):
        pass

    def _parse_geometries(self):
        pass


class ORCANBOOutput(MSONable):
    pass


class ORCATrajectoryOutput(MSONable):
    pass


class File47Output(MSONable):
    pass

