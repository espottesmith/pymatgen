"""
Input sets for ORCA
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import warnings
from typing import Literal

from monty.io import zopen

from pymatgen.io.orca.inputs import ORCAInput
from pymatgen.io.core import InputSet

from pymatgen.core.structure import Molecule

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2023, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"


class ORCASet(InputSet):
    """
    Class representing all inputs for an ORCA calculation
    """

    def __init__(
            self,
            orca_input: ORCAInput,
            optional_files: dict | None = None
    ):
        self.orca_input = orca_input
        self.optional_files = optional_files

    def write_input(
            self,
            directory: str | Path,
            make_dir: bool = True,
            overwrite: bool = True,
    ):
        """
        Write ORCA input file(s) to a directory

        Args:
            directory (str | Path): directory to write input files to
            make_dir (bool): If directory does not already exist, should it be created?
            overwrite (bool): If input file(s) already exist, should they be overwritten?
        """

        directory = Path(directory)

        if make_dir and not directory.exists():
            os.makedirs(directory)

        inputs = {
            "calc.inp": self.orca_input
        }
        if self.optional_files is not None:
            inputs.update(self.optional_files)

        for k, v in inputs.items():
            if v is not None and (overwrite or not (directory / k).exists()):
                if isinstance(v, ORCAInput):
                    v.write(directory / k)
                elif isinstance(v, Molecule):
                    v.to((directory / k).as_posix(), "xyz")
                else:
                    with zopen(directory / k, "wt") as f:
                        f.write(v.__str__())
            elif not overwrite and (directory / k).exists():
                raise FileExistsError(f"{directory / k} already exists")
            
    @staticmethod
    def from_directory(
        directory: str | Path,
        optional_files: dict | None = None,
        input_file_name: str | None = None
    ):
        """
        Load a set of ORCA inputs from a specified directory.

        Note that only the input file ("calc.inp" by default) will be read unless
        optional_files is specified.

        Args:
            directory (str | Path): Directory from which to read ORCA inputs.
            optional_files (dict | None): Optional files to read in addition to the 
                input file, in the form of {filename: object class}. 
        """
        pass