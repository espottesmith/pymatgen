"""Module defining ORCA generators."""

from __future__ import annotations

from enum import Enum
import glob
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
from monty.io import zopen
from monty.serialization import loadfn
from pkg_resources import resource_filename
from pymatgen.io.core import InputGenerator
from pymatgen.core.structure import Molecule
from pymatgen.io.orca.inputs import ORCAInput
from pymatgen.io.orca.outputs import ORCAOutput
from pymatgen.io.orca.sets import ORCASet


class ORCAApproximation(Enum):
    RI = "ri"
    RIJCOSX = "rijcosx"
    RIJK = "rijk"


@dataclass
class ORCAInputGenerator(InputGenerator):
    """
    Parameters:
    ------------

    dft_rung (int):
        Type of DFT exchange-correlation functional to use. Based on the rung chosen (1-5),
        a different default functional will be chosen:
            1 (LDA): VNW5
            2 (GGA): PBE
            3 (meta-GGA): B97M-V
            4 (hybrid): wB97M-V
            5 (double hybrid): wB97X-2
            other: no default will be applied (user must supply functional in user_settings)
    basis_set (str):
        Basis set to use. If not provided, then the minimally augmented split-valence basis
        ma-def2-SVP will be used. NOTE: if a custom basis set is desired, input "custom" (not case
        sensitive).
    approximation (str | ORCAApproximation):
        Which approximation, if any, should be used to accelerate SCF calculations?
    cpcm_settings (dict | None):
        Settings to use with the polarizable continuum model (PCM) or solvent model with density (SMD).
        A minimal input dictionary will contain two keys: "epsilon" for the solvent's relative dielectric
        constant and "refrac" for the solvent's index of refraction.
        For SMD, a minimal input dictionary will contain one key: "smdsolvent" for the name of the
        solvent. If the solvent of interest is not in ORCA's SMD database, additional parameters need
        to be provided, including "epsilon" and "refrac" as above for pcm_settings, as well as
        other SMD-specific parameters:
            soln: index of refraction at 293K
            soln25: index of refraction at 298K
            sola: Abraham's hydrogen bond acidity
            solb: Abraham's hydrogen bond basicity
            solg: relative macroscopic surface tension
            solc: aromaticity fraction
            solh: electronegative halogenicity
        Default is None, meaning that the calculation will be performed in vacuum
    user_settings (dict):
        Settings that override the default settings
    config_dict (dict):
        The config dictionary to use containing the base input set settings.
    """

    dft_rung: int | None = 4
    basis_set: str | None = "ma-def2-SVP"
    approximation: str | ORCAApproximation | None = None
    cpcm_settings: dict | None = None
    user_settings: dict = field(default_factory=dict)
    config_dict: dict = field(default_factory=dict)

    def get_input_set(  # type: ignore
        self,
        molecule: Molecule | None = None,
        prev_dir: str | Path | None = None,
        optional_files: dict | None = None,
        **kwargs
    ) -> ORCASet:
        """
        Get an ORCA input set.

        Note, if both ``molecule`` and ``prev_dir`` are set, then the molecule
        specified will be preferred over the final molecule from the last ORCA run.

        Args:
            molecule (Molecule | None)
            prev_dir (str | path | None): previous directory from which to generate the input set
            optional_files (dict | None): Additional files to be included in the input set.

        Returns:
            orca_set: ORCASet
        """
        molecule, prev_input = self._get_previous(molecule, prev_dir)

        input_updates = self.get_input_updates(
            molecule=molecule,
            prev_input=prev_input,
            **kwargs
        )

        orca_input = self._get_input(
            molecule,
            input_updates,
        )

        optional_files = optional_files if optional_files else dict()
        
        return ORCASet(orca_input=orca_input, optional_files=optional_files)

    def get_input_updates(
            self, **kwargs) -> dict:
        """
        Get updates to the ORCA input for this calculation type.

        Args:
            molecule (Molecule)
            prev_input (ORCAInput): ORCA input from a previous calculation

        Returns:
            dict
                A dictionary of updates to apply.
        """

        raise NotImplementedError

    
    def _get_previous(
            self,
            molecule: Molecule | None = None,
            prev_dir: str | Path | None = None,
            input_filename: str | None = None,
            output_filename: str | None = None
    ):
        """Load previous calculation outputs and decide which structure to use."""
        if molecule is None and prev_dir is None:
            raise ValueError("Either molecule or prev_dir must be set.")

        prev_input = dict()
        prev_structure = None

        if prev_dir:
            if input_filename is None:
                input_filename = "orca.inp"
            if output_filename is None:
                output_filename = "orca.out"
            prev_input = ORCAInput(Path(prev_dir) / input_filename)
            orca_output = ORCAOutput(Path(prev_dir) / output_filename)
            prev_structure = orca_output.data["molecule_from_final_geometry"]

        molecule = molecule if molecule is not None else prev_structure

        return molecule, prev_input, orca_output

    def _get_input(
        self,
        molecule: Molecule,
        input_updates: dict | None = None,
    ):
        """Get the input."""
        input_updates = dict() if input_updates is None else input_updates
        input_settings = dict(self.config_dict.get("orca_input", dict()))

        simple_input = list()

        # Add exchange-correlation functional to simple input section
        if self.dft_rung == 1:
            simple_input.append("VWN5")
        elif self.dft_rung == 2:
            simple_input.append("PBE")
        elif self.dft_rung == 3:
            simple_input.append("B97M-V")
        elif self.dft_rung == 4:
            simple_input.append("wB97M-V")
        elif self.dft_rung == 5:
            simple_input.append("wB97X-2")

        # Add basis set to simple input section
        if self.basis_set.lower() != "custom":
            simple_input.append(self.basis_set)

        if self.approximation is None:
            simple_input.append("NORI")
        else:
            if isinstance(approx, ORCAApproximation):
                approx = approx.value
                
            if approx == "ri":
                simple_input.append("RI")
            elif approx == "rijcosx":
                simple_input.append("RIJCOSX")
            elif approx == "rijk":
                simple_input.append("RI-JK")

        # Always use tight grid by default
        simple_input.append("defgrid3")

        if self.pcm_settings is not None and self.smd_settings is not None:
            raise ValueError("Cannot provide settings for PCM and SMD!")
        elif self.cpcm_settings is not None:
            simple_input.append("CPCM")

        # Generate base input but override with user input settings
        input_settings = update_inputs(input_settings, input_updates)
        input_settings = update_inputs(input_settings, self.user_settings["orca_input"])
        
        if self.cpcm_settings is not None:
            input_settings["cpcm"] = update_inputs(input_settings["cpcm"], self.pcm_settings)

        if "simple_input" not in input_settings:
            input_settings["simple_input"] = simple_input
        else:
            si = input_settings["simple_input"]
            if isinstance(si, str):
                si = si.split()
            for this_si in simple_input:
                if this_si not in si:
                    # TODO: should have some kind of validation
                    # Right now, we simply assume that there are no conflicting options in the input
                    si.insert(0, this_si)

        input_settings["molecule"] = molecule

        return ORCAInput(**input_settings)


@dataclass
class SinglePointGenerator(ORCAInputGenerator):
    """
    Class to generate ORCA single-point energy input sets.
    """

    def get_input_updates(self, **kwargs) -> dict:
        """Get updates to the input for a single-point energy calculation"""

        return {"scf": {"maxiter": 200, "thresh": "1e-14", "convergence": "VeryTightSCF"}}
    

@dataclass
class NBOGenerator(ORCAInputGenerator):
    """
    Class to generate ORCA input sets using the natural bonding orbital (NBO) external program.
    """

    def get_input_updates(self, **kwargs) -> dict:
        """
        Get updates to the input for a single-point energy calculation using the natural bonding orbital
        (NBO) program
        """

        return {"scf": {"maxiter": 200, "thresh": "1e-14", "convergence": "VeryTightSCF"},
                "simple_input": ["NBO"]}


@dataclass
class GradientGenerator(ORCAInputGenerator):
    """
    Class to generate ORCA engrad (energy-gradient or energy-force) input sets.
    """
    
    def get_input_updates(self, analytic: bool = True, **kwargs) -> dict:
        """
        Get updates to the input for an engrad (energy-gradient) calculation
        """

        if analytic:
            runtyp = "engrad"
        else:
            runtyp = "numgrad"

        return {"scf": {"maxiter": 200, "thresh": "1e-14", "convergence": "VeryTightSCF"},
                "method": {"runtyp": runtyp}}
    

@dataclass
class MinOptGenerator(ORCAInputGenerator):
    """
    Class to generate ORCA geometry optimization input sets for optimization to PES minima
    """

    def get_input_updates(
            self,
            max_opt_iters: int = 100,
            max_step: float = 0.3,
            trust_radius: float = -0.3,
            **kwargs
        ):
        """
        Get updates to the input for a geometry optimization calculation

        Args:
            max_opt_iters (int): Maximum number of optimization steps (default 100)
            max_step (float): Maximum step size in a.u. (default 100)
            trust_radius (float): Trust radius in a.u. (default -0.3, meaning a fixed radius
            of 0.3)
        """

        return {
            "scf": {"maxiter": 200, "convergence": "VeryTightSCF"},
            "method": {"runtyp": "opt"},
            "geom": {
                "maxiter": max_opt_iters,
                "coordsys": "redundant_new",
                "maxstep": max_step,
                "trust_radius": trust_radius
                }
        }
    

@dataclass
class OptFreqGenerator(ORCAInputGenerator):
    """
    Class to generate ORCA input sets for geometry optimization calculations followed
    by vibrational frequency analysis
    """

    def get_input_updates(
            self,
            max_opt_iters: int = 100,
            max_step: float = 0.3,
            trust_radius: float = -0.3,
            analytic: bool = True,
            scaling_factor: float = 1.0,
            **kwargs
        ):
        """
        Get updates to the input for a geometry optimization calculation followed by vibrational
        frequency analysis

        Args:
            max_opt_iters (int): Maximum number of optimization steps (default 100)
            max_step (float): Maximum step size in a.u. (default 100)
            trust_radius (float): Trust radius in a.u. (default -0.3, meaning a fixed radius
                of 0.3)
            analytic (bool): If True (default), calculate frequencies analytically. This should
                be possible for most DFT functionals in ORCA.
            scaling_factor (float): Scaling factor for frequencies, applied after the frequencies
                have been calculated (default 1.0)
        """

        if analytic:
            freq = {"anfreq": "true"}
        else:
            freq = {"anfreq": "false"}

        freq["scalfreq"] = scaling_factor

        return {
            "scf": {"maxiter": 200, "convergence": "VeryTightSCF"},
            "method": {"runtyp": "opt"},
            "geom": {
                "maxiter": max_opt_iters,
                "coordsys": "redundant_new",
                "maxstep": max_step,
                "trust_radius": trust_radius
                },
            "freq": freq
        }


@dataclass
class TransitionStateOptGenerator(ORCAInputGenerator):
    """
    Class to generate ORCA transition-state geometry optimization input sets
    """

    def get_input_updates(
        max_opt_iters: int = 100,
        max_step: float = 0.3,
        trust_radius: float = -0.1,
        **kwargs
    ):
        """
        Get updates to the input for a transition-state optimization calculation

        Args:
            max_opt_iters (int): Maximum number of optimization steps (default 100)
            max_step (float): Maximum step size in a.u. (default 100)
            trust_radius (float): Trust radius in a.u. (default -0.3, meaning a fixed radius
            of 0.3)
        """

        return {
            "scf": {"maxiter": 200, "convergence": "VeryTightSCF"},
            "method": {"runtyp": "opt"},
            "geom": {
                "maxiter": max_opt_iters,
                "coordsys": "redundant_new",
                "maxstep": max_step,
                "trust_radius": trust_radius,
                "ts_search": "EF",
                "calc_hess": "true"
                }
        }
    

@dataclass
class TSOptFreqGenerator(ORCAInputGenerator):
    """
    Class to generate ORCA input sets for calculations involving transition-state optimization
    followed by vibrational frequency analysis.
    """

    def get_input_updates(
        max_opt_iters: int = 100,
        max_step: float = 0.3,
        trust_radius: float = -0.1,
        analytic: bool = True,
        scaling_factor: float = 1.0,
        **kwargs
    ):
        """
        Get updates to the input for a transition-state optimization calculation

        Args:
            max_opt_iters (int): Maximum number of optimization steps (default 100)
            max_step (float): Maximum step size in a.u. (default 100)
            trust_radius (float): Trust radius in a.u. (default -0.3, meaning a fixed radius
            of 0.3)
            analytic (bool): If True (default), calculate frequencies analytically. This should
                be possible for most DFT functionals in ORCA.
            scaling_factor (float): Scaling factor for frequencies, applied after the frequencies
                have been calculated (default 1.0)
        """

        if analytic:
            freq = {"anfreq": "true"}
        else:
            freq = {"anfreq": "false"}

        freq["scalfreq"] = scaling_factor

        return {
            "scf": {"maxiter": 200, "convergence": "VeryTightSCF"},
            "method": {"runtyp": "opt"},
            "geom": {
                "maxiter": max_opt_iters,
                "coordsys": "redundant_new",
                "maxstep": max_step,
                "trust_radius": trust_radius,
                "ts_search": "EF",
                "calc_hess": "true"
                },
            "freq": freq
        }
    

@dataclass
class PESScanGenerator(ORCAInputGenerator):
    """
    Class to generate ORCA input sets for calculations involving potential energy surface (PES) scans.
    """
    
    def get_input_updates(
            self,
            max_opt_iters: int = 100,
            max_step: float = 0.3,
            trust_radius: float = -0.3,
            scan_variables: List[Dict[str, Any]] | None = None,
            **kwargs
        ):
        """
        Get updates to the input for a geometry optimization calculation

        Args:
            max_opt_iters (int): Maximum number of optimization steps (default 100)
            max_step (float): Maximum step size in a.u. (default 100)
            trust_radius (float): Trust radius in a.u. (default -0.3, meaning a fixed radius
            of 0.3)
            scan_variables (List[Dict[str, Any]] | None): PES scan variables, given in the format
                {"type": "B" | "A" | "D",
                 "atoms": List[int],
                 "start": float,
                 "end": float,
                 "num_steps": int}
        """

        # For a PES scan job, need to have scan variables present
        if scan_variables is None:
            raise ValueError("No scan_variables provided for PES scan job!")
        elif len(scan_variables) == 0:
            raise ValueError("No scan_variables provided for PES scan job!")

        geom = {
                "maxiter": max_opt_iters,
                "coordsys": "redundant_new",
                "maxstep": max_step,
                "trust_radius": trust_radius
                }
        
        scan = list()
        for sv in scan_variables:
            atoms = " ".join(sv.get("atoms", list()))
            string = f'{sv.get("type")} {atoms} = {sv.get("start")}, {sv.get("end")}, {sv.get("num_steps")}'
            scan.append(string)

        # Format as a string
        geom["scan"] = "\n      ".join(scan)

        return {
            "scf": {"maxiter": 200, "convergence": "VeryTightSCF"},
            "method": {"runtyp": "opt"},
            "geom": geom
        }

@dataclass
class NEBGenerator(ORCAInputGenerator):
    """
    Class to generate ORCA input sets for calculations involving nudged elastic band (NEB) calculations
    for transition-state searching.
    """

    def get_input_updates(
            self,
            method: str = "NEB-TS",
            endpoint_filename: str = "endpoint.xyz",
            guess_filename: str | None = None,
            **kwargs) -> dict:
        """
        
        
        """
        
        if method.upper() not in ["NEB-TS", "NEB-CI", "ZOOM-NEB-TS", "ZOOM-NEB-CI",
                                  "FAST-NEB-TS", "LOOSE-NEB-TS"]:
            raise ValueError("Inappropriate NEB method! Acceptable values are "
                             "NEB-TS, NEB-CI, ZOOM-NEB-TS, ZOOM-NEB-CI, FAST-NEB-TS, and LOOSE-NEB-TS")

        # TODO: should figure out some way to verify if the specified files exist

        neb = {"neb_end_xyzfile": endpoint_filename}
        if guess_filename is not None:
            neb["neb_ts_xyzfile"] = guess_filename

        return {
            "simple_input": [method],
            "scf": {"maxiter": 200, "convergence": "VeryTightSCF"},
            "neb": neb
        } 


@dataclass
class FrequencyGenerator(ORCAInputGenerator):
    """
    Class to generate ORCA input sets for calculations involving vibrational frequency analyses.
    """

    def get_input_updates(
        analytic: bool = True,
        scaling_factor: float = 1.0,
        temperatures: float | List[float] = 298.15
    ):
        """
        Get updates to the input for a frequency calculation

        Args:
            analytic (bool): If True (default), calculate frequencies analytically. This should
                be possible for most DFT functionals in ORCA.
            scaling_factor (float): Scaling factor for frequencies, applied after the frequencies
                have been calculated (default 1.0)
            temperatures (float | List[float]): Temperatures at which thermodynamics (e.g. Gibbs free energy)
                will be evaluated
        """

        if analytic:
            freq = {"anfreq": "true"}
        else:
            freq = {"anfreq": "false"}

        freq["scalfreq"] = scaling_factor

        if isinstance(temperatures, list):
            freq["temp"] = ", ".join([str(x) for x in temperatures])
        else:
            freq["temp"] = temperatures

        return {
            "scf": {"maxiter": 200, "convergence": "VeryTightSCF"},
            "method": {"runtyp": "opt"},
            "freq": freq
        }


def update_inputs(d: dict, u: dict):
    """
    Update a dictionary recursively and return it.

    Parameters
    ----------
        d: Dict
            Input dictionary to modify
        u: Dict
            Dictionary of updates to apply

    Returns
    -------
    Dict
        The updated dictionary.

    Example
    ----------
        d = {'activate_hybrid': {"hybrid_functional": "HSE06"}}
        u = {'activate_hybrid': {"cutoff_radius": 8}}

        yields {'activate_hybrid': {"hybrid_functional": "HSE06", "cutoff_radius": 8}}}
    """
    for k, v in u.items():
        if k == "simple_input":
            if d.get(k) is None:
                d[k] = list()
            elif isinstance(d.get(k), str):
                d[k] = d[k].strip().split()
            
            if isinstance(v, str):
                v = v.strip().split()
            for iv in v:
                if iv not in d[k]:
                    d[k].append(iv)
        else:
            if isinstance(v, dict):
                d[k] = update_inputs(d.get(k, {}), v)
            else:
                d[k] = v
    return d