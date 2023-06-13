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
from typing import TYPE_CHECKING, Any, List

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
    basis_set (str):
        Basis set to use. If not provided, then the minimally augmented split-valence basis
        ma-def2-SVP will be used
    approximations (List[str | ORCAApproximation]):
        Which approximations, if any, should be used to accelerate SCF calculations?
    pcm_settings (dict | None):
        Settings to use with the polarizable continuum model (PCM). A minimal input dictionary will
        contain two keys: "epsilon" for the solvent's relative dielectric constant and "refrac" for
        the solvent's index of refraction.
        Default is None, meaning that the calculation will be performed in vacuum
    smd_settings (dict | None):
        Settings to use with the solvent model with density (SMD). A minimal input dictionary will
        contain one key: "smdsolvent" for the name of the solvent. If the solvent of interest is not in
        ORCA's SMD database, additional parameters need to be provided, including "epsilon" and "refrac"
        as above for pcm_settings, as well as other SMD-specific parameters:
            soln: index of refraction at 293K
            soln25: index of refraction at 298K
            sola: Abraham's hydrogen bond acidity
            solb: Abraham's hydrogen bond basicity
            solg: relative macroscopic surface tension
            solc: aromaticity fraction
            solh: electronegative halogenicity
    user_settings (dict):
        Settings that override the default settings
    config_dict (dict):
        The config dictionary to use containing the base input set settings.
    """

    dft_rung: int = 4
    basis_set: str = "ma-def2-SVP"
    approximations: List[str | ORCAApproximation] = field(default_factory=list)
    pcm_settings: dict | None = None
    smd_settings: dict | None = None
    user_settings: dict = field(default_factory=dict)
    config_dict: dict = field(default_factory=dict)

    def get_input_set(  # type: ignore
        self,
        molecule: Molecule | None = None,
        prev_dir: str | Path | None = None,
        optional_files: dict | None = None,
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
            molecule,
            prev_input=prev_input,
        )

        orca_input = self._get_input(
            molecule,
            prev_input,
            input_updates,
        )

        optional_files = optional_files if optional_files else dict()
        
        return ORCASet(orca_input=orca_input, optional_files=optional_files)

    def get_input_updates(
            self,
            molecule: Molecule,
            prev_input: ORCAInput) -> dict:
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
        previous_input: ORCAInput | None = None,
        input_updates: dict | None = None,
    ):
        """Get the input."""
        previous_input = dict() if previous_input is None else previous_input
        input_updates = dict() if input_updates is None else input_updates
        input_settings = dict(self.config_dict["orca_input"])

        # Generate base input but override with user input settings
        input_settings = recursive_update(input_settings, input_updates)
        input_settings = recursive_update(input_settings, self.user_input_settings)
        overrides = (
            input_settings.pop("override_default_params")
            if "override_default_params" in input_settings
            else {}
        )
        cp2k_input = DftSet(structure=structure, kpoints=kpoints, **input_settings)

        for setting in input_settings:
            if (
                hasattr(cp2k_input, setting)
                and input_settings[setting]
                and callable(getattr(cp2k_input, setting))
            ):
                subsettings = input_settings.get(setting)
                getattr(cp2k_input, setting)(
                    **subsettings if isinstance(subsettings, dict) else {}
                )

        cp2k_input.update(overrides)
        return cp2k_input



def recursive_update(d: dict, u: dict):
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
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d