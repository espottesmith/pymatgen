"""
Classes for reading/manipulating/writing ORCA input files.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from monty.io import zopen

from pymatgen.core import Molecule
from pymatgen.io.core import InputFile

from pymatgen.io.qchem.utils import read_pattern, read_table_pattern
from pymatgen.io.orca.utils import check_unique_block

if TYPE_CHECKING:
    from pathlib import Path

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2023, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__credits__ = "Alexander Epstein, Sam Blau"

logger = logging.getLogger(__name__)


class ORCAInput(InputFile):
    """
    An object representing an ORCA input file. In most cases, ORCAInput attributes represent different "blocks" of an
    ORCA input file. Not all blocks are included in ORCAInput at this time. To add a new block, one needs to modify
    __init__, __str__, from_sting and add static methods to read and write the new block, i.e. section_template and
    read_section. By design, there is very little (or no) checking that input parameters conform to the appropriate ORCA
    format, this responsibility lands on the user or a separate error handling software.
    """

    def __init__(
            self,
            molecule: Molecule,
            simple_input: Optional[Union[str, List[str]]] = None,
            basis: Optional[Dict[str, Any]] = None, 
            cpcm: Optional[Dict[str, Any]] = None,
            freq: Optional[Dict[str, Any]] = None,
            geom: Optional[Dict[str, Any]] = None,
            method: Optional[Dict[str, Any]] = None,
            nbo: Optional[Dict[str, Any]] = None,
            numgrad: Optional[Dict[str, Any]] = None,
            output: Optional[Dict[str, Any]] = None,
            pal: Optional[Dict[str, Any]] = None,
            plots: Optional[Dict[str, Any]] = None,
            scf: Optional[Dict[str, Any]] = None,
            ):
        
        """
        Args:
            molecule (Molecule): Input structure.
            simple_input (Optional[Union[str, List[str]]]): Simple input string. If this is None, then simple input will
                not be used, and all input (inlcuding basis and method) will be described in blocks
            basis (Optional[Dict[str, Any]]): key-value pairs for %basis block
            cpcm (Optional[Dict[str, Any]]): key-value pairs for %cpcm block
            freq (Optional[Dict[str, Any]]): key-value pairs for %freq block
            geom (Optional[Dict[str, Any]]): key-value pairs for %geom block
            method (Optional[Dict[str, Any]]): key-value pairs for %method block
            nbo (Optional[Dict[str, Any]]): key-value pairs for %nbo block
            numgrad (Optional[Dict[str, Any]]): key-value pairs for %numgrad block
            output (Optional[Dict[str, Any]]): key-value pairs for %output block
            pal (Optional[Dict[str, Any]]): key-value pairs for %pal block
            plots (Optional[Dict[str, Any]]): key-value pairs for %plots block
            scf (Optional[Dict[str, Any]]): key-value pairs for %scf block
        """

        self.molecule = molecule
        self.simple_input = simple_input
        self.basis = check_unique_block(basis)
        self.cpcm = check_unique_block(cpcm)
        self.freq = check_unique_block(freq)
        self.geom = check_unique_block(geom)
        self.method = check_unique_block(method)
        self.nbo = check_unique_block(nbo)
        self.numgrad = check_unique_block(numgrad)
        self.output = check_unique_block(output)
        self.pal = check_unique_block(pal)
        self.plots = check_unique_block(plots)
        self.scf = check_unique_block(scf)

    def get_string(self):
        """
        Return a string representation of an entire input file.
        """
        return str(self)
    
    def __str__(self):
        combined_list = []
        
        # Simple input line
        if self.simple_input:
            if isinstance(self.simple_input, str):
                combined_list.append("! " + self.simple_input)
            else:
                combined_list.append("! " + " ".join(self.simple_input))
            combined_list.append("")

        # molecule block
        combined_list.append(self.molecule_template(self.molecule))
        combined_list.append("")

        # basis block
        if self.basis:
            combined_list.append(self.template(self.basis, "basis"))
            combined_list.append("")
        # cpcm block
        if self.cpcm:
            combined_list.append(self.template(self.cpcm, "cpcm"))
            combined_list.append("")
        # freq block
        if self.freq:
            combined_list.append(self.template(self.freq, "freq"))
            combined_list.append("")
        # geom block
        if self.geom:
            combined_list.append(self.template(self.geom, "geom"))
            combined_list.append("")
        # method block
        if self.method:
            combined_list.append(self.template(self.method, "method"))
            combined_list.append("")
        # nbo block
        if self.nbo:
            combined_list.append(self.template(self.nbo, "nbo"))
            combined_list.append("")
        # numgrad block
        if self.numgrad:
            combined_list.append(self.template(self.numgrad, "numgrad"))
            combined_list.append("")
        # output section
        if self.output:
            combined_list.append(self.template(self.output, "output"))
            combined_list.append("")
        # pal section
        if self.pal:
            combined_list.append(self.template(self.pal, "pal"))
            combined_list.append("")
        # plots section
        if self.plots:
            combined_list.append(self.template(self.plots, "plots"))
            combined_list.append("")
        # scf section
        if self.scf:
            combined_list.append(self.template(self.scf, "scf"))
            combined_list.append("")
        return "\n".join(combined_list)
    
    @classmethod
    def from_string(cls, string: str) -> ORCAInput:
        """
        Construct ORCAInput from string.

        Args:
            string (str): String input.

        Returns:
            ORCAInput
        """
        blocks = cls.find_blocks(string)
        molecule = cls.read_molecule(string)
        simple_input = None
        basis = None
        cpcm = None
        freq = None
        geom = None
        method = None
        nbo = None
        numgrad = None
        output = None
        pal = None
        plots = None
        scf = None

        if "simple_input" in blocks:
            simple_input = cls.read_simple_input(string)
        if "basis" in blocks:
            basis = cls.read_block(string, "basis")
        if "cpcm" in blocks:
            cpcm = cls.read_block(string, "cpcm")
        if "freq" in blocks:
            freq = cls.read_block(string, "freq")
        if "geom" in blocks:
            geom = cls.read_block(string, "geom")
        if "method" in blocks:
            method = cls.read_block(string, "method")
        if "nbo" in blocks:
            nbo = cls.read_block(string, "nbo")
        if "numgrad" in blocks:
            numgrad = cls.read_block(string, "numgrad")
        if "output" in blocks:
            output = cls.read_block(string, "output")
        if "pal" in blocks:
            pal = cls.read_block(string, "pal")
        if "plots" in blocks:
            plots = cls.read_block(string, "plots")
        if "scf" in blocks:
            scf = cls.read_block(string, "scf")

        return cls(
            molecule=molecule,
            simple_input=simple_input,
            basis=basis,
            cpcm=cpcm,
            freq=freq,
            geom=geom,
            method=method,
            nbo=nbo,
            numgrad=numgrad,
            output=output,
            pal=pal,
            plots=plots,
            scf=scf,
        )
    
    @staticmethod
    def from_file(filename: str | Path) -> ORCAInput:
        """
        Create an ORCAInput from file.

        Args:
            filename (str): Filename

        Returns:
            ORCAInput
        """
        with zopen(filename, "rt") as f:
            return ORCAInput.from_string(f.read())
    
    @classmethod
    def molecule_template(cls, molecule: Molecule) -> str:
        """
        Template for a molecule in an ORCA input file

        Args:
            molecule (Molecule)

        Returns:
            (str) Molecule input for ORCA.
        """

        mol_list = [f"* xyz {int(molecule.charge)} {int(molecule.spin_multiplicity)}"]
        for site in molecule.sites:
            mol_list.append(f" {site.species_string}     {site.x: .10f}     {site.y: .10f}     {site.z: .10f}")

        mol_list.append("*")
        return "\n".join(mol_list)

    @classmethod
    def template(cls, contents: Dict[str, Any], block_name: str) -> str:
        """
        Generic template for an ORCA input block

        Args:
            contents (Dict[str, Any]): Key-value pairs of inputs
            block_name (str): Name of this block

        Returns:
            (str) Block for an ORCA input file
        """

        block_list = [f"%{block_name}"]
        for k, v in contents.items():
            if isinstance(v, list):
                block_list.append(f"  {k} {','.join(v)}")
            else:
                block_list.append(f"  {k} {v}")
        block_list.append("end")
        return "\n".join(block_list)

    @staticmethod
    def find_blocks(string: str) -> List[str]:
        """
        Identify blocks in an ORCA input file.

        Args:
            string (str): string (representation of an input file) to be searched

        Returns:
            (List[str]) List of block names
        """

        patterns = {"blocks": r"^%([a-z_]+)"}
        matches = read_pattern(string, patterns)

        # list of the blocks present
        blocks = [val[0] for val in matches["blocks"]]
        return blocks

    @staticmethod
    def read_molecule(string: str) -> Molecule:
        """
        Read a molecule from the molecule section of an ORCA input file.

        Args:
            string (str): string (representation of an input file) to be searched

        Returns:
            Molecule
        """

        charge = None
        spin_mult = None
        patterns = {
            "xyzfile": r"^\s*\* xyzfile ((?:\-)*\d+)\s+((?:\-)*\d+)\s+([A-Za-z0-9\-\_\.]+)",
            "charge": r"^\s*\*xyz\s+([0-9\-]+)\s+[0-9\-]+",
            "spin_mult": r"^\s*\*xyz\s+[0-9\-]+\s+([0-9\-]+)",
        }
        matches = read_pattern(string, patterns)
        if "xyzfile" in matches:
            charge = int(matches["xyzfile"][0][0])
            spin_mult = int(matches["xyzfile"][0][1])
            try:
                mol = Molecule.from_file(matches["xyzfile"][0][2])
                mol.set_charge_and_spin(charge, spin_multiplicity=spin_mult)
                return mol
            except FileNotFoundError:
                raise Exception("Cannot build ORCAInput: molecule provided as XYZ file, but file does not exist!")

        if "charge" in matches:
            charge = float(matches["charge"][0][0])
        if "spin_mult" in matches:
            spin_mult = int(matches["spin_mult"][0][0])

        header = r"^\s*\*xyz\s+[0-9\-]+\s+[0-9\-]+"
        row = r"\s*([A-Za-z]+)\s+([\d\-\.]+)\s+([\d\-\.]+)\s+([\d\-\.]+)"
        footer = r"^\s*\*"
        mol_table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
        species = [val[0] for val in mol_table[0]]
        coords = [[float(val[1]), float(val[2]), float(val[3])] for val in mol_table[0]]
        if charge is None:
            mol = Molecule(species=species, coords=coords)
        else:
            mol = Molecule(species=species, coords=coords, charge=charge, spin_multiplicity=spin_mult)
        return mol

    @staticmethod
    def read_simple_input(string: str) -> List[str]:
        """
        Parse simple input from string.

        Args:
            string (str): string (representation of an input file) to be searched

        Returns:
            (List[str]) list representation of simple input string
        """

        patterns = {"simple_input": r"^\s*?!\s*([^\n]+)"}
        matches = read_pattern(string, patterns)

        if "simple_input" in matches:
            # Break it down as a list, separated by whitespace
            simple_input = matches["simple_input"][0][0].strip().split()
            return simple_input
        else:
            return None
    
    @staticmethod
    def read_block(string: str, block_name: str) -> Dict[str, Any]:
        """
        Parse a ORCA input block

        Args:
            string (str): string (representation of an input file) to be searched
            block_name (str): Name of the block to search for

        Returns:
            (Dict[str, Any]])
        """

        header = r"^%" + block_name
        row = r"\s*([a-zA-Z\_\d]+)\s+([a-zA-Z\-\_\d]+)"
        footer = r"^end"

        table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
        return dict(table[0])