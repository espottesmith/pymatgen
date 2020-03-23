# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import logging
from typing import Dict, List, Tuple, Optional, Union, Iterator, Set, Sequence, Iterable

import numpy as np

from monty.json import MSONable
from monty.io import zopen

from pymatgen.core import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.qchem.utils import (read_table_pattern,
                                     read_pattern,
                                     lower_and_check_unique)

# Classes for reading/manipulating/writing Q-Chem input files.

__author__ = "Brandon Wood, Samuel Blau, Shyam Dwaraknath, Julian Self, " \
             "Evan Spotte-Smith"
__copyright__ = "Copyright 2018, The Materials Project"
__version__ = "0.1"
__email__ = "b.wood@berkeley.edu"
__credits__ = "Xiaohui Qu"

logger = logging.getLogger(__name__)


class QCInput(MSONable):
    """
    An object representing a Q-Chem input file. QCInput attributes represent
    different sections of a Q-Chem input file.

    To add a new section one needs to modify __init__, __str__, from_sting and
    add staticmethods to read and write the new section i.e. section_template
    and read_section. By design, there is very little (or no) checking that
    input parameters conform to the appropriate Q-Chem format, this
    responsibility lands on the user or a separate error handling software.

    Args:
        molecule (pymatgen Molecule object or "read"):
            Input molecule. molecule can be set as either a pymatgen Molecule
            object or as the str "read". "read" can be used in multi_job Q-Chem
            input files where the molecule is read in from the previous
            calculation.
        rem (dict):
            A dictionary of all the input parameters for the rem section of
            the Q-Chem input file.
            Ex. rem = {'method': 'rimp2', 'basis': '6-31*G++' ... }
        opt (dict of lists):
            A dictionary of opt sections, where each opt section is a key and
            the corresponding values are a list of strings. Stings must be
            formatted as instructed by the Q-Chem manual. The different opt
            sections are: CONSTRAINT, FIXED, DUMMY, and CONNECT
            Ex. opt = {"CONSTRAINT": ["tors 2 3 4 5 25.0", "tors 2 5 7 9 80.0"],
                       "FIXED": ["2 XY"]}
        pcm (dict):
            A dictionary of all the input parameters for the pcm input section
            of the Q-Chem input file. Note that other implicit solvent methods
            (ex: SMD) do not require the use of the pcm section. Further note
            that this section will not be read unless a "solvent_method" key is
            included in the rem section, with the value "pcm"
            Ex. pcm = {"theory": "cosmo", "printlevel": 2}
        solvent (dict):
            A dictionary specifying the solvent used in the PCM implicit solvent
            method. This section will not be read unless a "solvent_method" key
            is included in the rem section.
            Ex. solvent = {"dielectric": 78.4, "opticaldielectric": 1.78}
        smx (dict):
            A dictionary specifying the input parameters for any SMx (SM8, SM12,
            SMD) implicit solvent method in the Q-Chem input file. Note that
            other implicit solvent methods (ex: PCM) do not require the use of
            the smx section. Further note that this section will not be read
            unless a "solvent_method" key is included in the rem section with
            the value of "sm8", "sm12", or "smd".
            Ex. smx = {"solvent": "water"}
        scan (dict of lists):
            A dictionary of scan variables. Because two constraints of the same
            type are allowed (for instance, two torsions or two bond stretches),
            each TYPE of variable (stre, bend, tors) should be its own key in
            the dict, rather than each variable. Note that the total number of
            variable (sum of lengths of all lists) CANNOT be more than two.
            Ex. scan = {"stre": ["3 6 1.5 1.9 0.1"],
                        "tors": ["1 2 3 4 -180 180 15"]}
    """

    def __init__(self, molecule, rem, opt=None, pcm=None, solvent=None,
                 smx=None, scan=None, plots=None):
        self.molecule = molecule
        self.rem = lower_and_check_unique(rem)
        self.opt = opt
        self.pcm = lower_and_check_unique(pcm)
        self.solvent = lower_and_check_unique(solvent)
        self.smx = lower_and_check_unique(smx)
        self.scan = lower_and_check_unique(scan)
        self.plots = lower_and_check_unique(plots)

        # Make sure molecule is valid
        if isinstance(self.molecule, str):
            self.molecule = self.molecule.lower()
            if self.molecule != "read":
                raise ValueError('The only acceptable text value for molecule '
                                 'is "read"')

        elif not isinstance(self.molecule, Molecule):
            raise ValueError("The molecule must either be the string 'read' or"
                             " a pymatgen Molecule object.")

        # Make sure rem is valid:
        #   - Has a basis
        #   - Has a method or DFT exchange functional
        #   - Has a valid job_type or jobtype
        valid_job_types = ["opt", "optimization", "sp", "freq", "frequency",
                           "force", "nmr", "ts", "pes_scan"]

        if "basis" not in self.rem:
            raise ValueError("The rem dictionary must contain a 'basis' entry")
        if "method" not in self.rem:
            if "exchange" not in self.rem:
                raise ValueError(
                    "The rem dictionary must contain either a 'method' entry or an 'exchange' entry"
                )
        if "job_type" not in self.rem:
            raise ValueError(
                "The rem dictionary must contain a 'job_type' entry")
        if self.rem.get("job_type").lower() not in valid_job_types:
            raise ValueError(
                "The rem dictionary must contain a valid 'job_type' entry")

        #TODO:
        #   - Check that the method or functional is valid
        #   - Check that basis is valid
        #   - Check that basis is defined for all species in the molecule
        #   - Validity checks specific to job type?
        #   - Check OPT and PCM sections?

    def __str__(self):
        combined_list = list()
        # molecule section
        combined_list.append(self.molecule_template(self.molecule))
        combined_list.append("")
        # rem section
        combined_list.append(self.rem_template(self.rem))
        combined_list.append("")
        # opt section
        if self.opt:
            combined_list.append(self.opt_template(self.opt))
            combined_list.append("")
        # pcm section
        if self.pcm:
            combined_list.append(self.pcm_template(self.pcm))
            combined_list.append("")
        # solvent section
        if self.solvent:
            combined_list.append(self.solvent_template(self.solvent))
            combined_list.append("")
        if self.smx:
            combined_list.append(self.smx_template(self.smx))
            combined_list.append("")
        if self.scan:
            combined_list.append(self.scan_template(self.scan))
            combined_list.append("")
        # plots section
        if self.plots:
            combined_list.append(self.plots_template(self.plots))
            combined_list.append("")
        return '\n'.join(combined_list)

    @staticmethod
    def multi_job_string(job_list):
        """
        Convert a list of Q-Chem jobs into a string so that the jobs can be run
        sequentially.

        Args:
            job_list (list of QCInput objects): List of jobs to be converted to
                input file text.
        Returns:
            multi_job_string (str): Text of a Q-Chem input file to run the job
                list.
        """
        multi_job_string = str()
        for i, job_i in enumerate(job_list):
            if i < len(job_list) - 1:
                multi_job_string += job_i.__str__() + "\n@@@\n\n"
            else:
                multi_job_string += job_i.__str__()
        return multi_job_string

    @classmethod
    def from_string(cls, string):
        """
        Convert a string into a QCInput object.

        Args:
            string (str): Text of a Q-Chem input file.
        Returns:
            QCInput
        """
        sections = cls.find_sections(string)
        molecule = cls.read_molecule(string)
        rem = cls.read_rem(string)

        # only molecule and rem are necessary
        # All other sections are checked as needed
        opt = None
        pcm = None
        solvent = None
        smx = None
        scan = None
        plots = None

        if "opt" in sections:
            opt = cls.read_opt(string)
        if "pcm" in sections:
            pcm = cls.read_pcm(string)
        if "solvent" in sections:
            solvent = cls.read_solvent(string)
        if "smx" in sections:
            smx = cls.read_smx(string)
        if "scan" in sections:
            scan = cls.read_scan(string)
        if "plots" in sections:
            plots = cls.read_plots(string)

        return cls(molecule, rem, opt=opt, pcm=pcm, solvent=solvent, smx=smx, scan=scan, plots=plots)

    def write_file(self, filename):
        """
        Write the QCInput object as a Q-Chem input file.

        Recommended file suffix for Q-Chem input files: *.inp or *.qin

        Args:
            filename (str): File name or path to the Q-Chem input file.
        Returns:
            None
        """
        with zopen(filename, 'wt') as f:
            f.write(self.__str__())

    @staticmethod
    def write_multi_job_file(job_list, filename):
        """
        Write a list of QCInput objects as a single Q-Chem input file.

        Args:
            job_list (list of QCInput objects): List of jobs to be converted to
                input file text.
            filename (str): File name or path to the Q-Chem input file.
        Returns:
            None
        """
        with zopen(filename, 'wt') as f:
            f.write(QCInput.multi_job_string(job_list))

    @staticmethod
    def from_file(filename):
        """
        Read a Q-Chem input file and convert it to a QCInput object.

        Args:
            filename (str): File name or path to the Q-Chem input file.
        Returns:
            QCInput
        """
        with zopen(filename, 'rt') as f:
            return QCInput.from_string(f.read())

    @classmethod
    def from_multi_jobs_file(cls, filename):
        """
        Read a Q-Chem input file containing multiple jobs and convert it to a
            list of QCInput objects.

        Args:
            filename (str): File name or path to the Q-Chem input file.
        Returns:
            input_list (list of QCInput objects): Input objects, one per job
                in the Q-Chem input file.
        """
        # returns a list of QCInput objects
        with zopen(filename, 'rt') as f:
            # the delimiter between Q-Chem jobs is @@@
            multi_job_strings = f.read().split("@@@")
            # list of individual Q-Chem jobs
            input_list = [cls.from_string(i) for i in multi_job_strings]
            return input_list

    @staticmethod
    def molecule_template(molecule):
        """
        Provides the text for the molecule section of a Q-Chem input file.

        Args:
            molecule (Molecule or str): Molecule to be written.
        Return:
            str
        """

        #TODO: add ghost atoms
        mol_list = list()
        mol_list.append("$molecule")
        if isinstance(molecule, str):
            if molecule == "read":
                mol_list.append(" read")
            else:
                raise ValueError('The only acceptable text value for molecule is "read"')
        else:
            mol_list.append(" {charge} {spin_mult}".format(
                charge=int(molecule.charge),
                spin_mult=molecule.spin_multiplicity))
            for site in molecule.sites:
                mol_list.append(
                    " {atom}     {x: .10f}     {y: .10f}     {z: .10f}".format(
                        atom=site.species_string, x=site.x, y=site.y,
                        z=site.z))
        mol_list.append("$end")
        return '\n'.join(mol_list)

    @staticmethod
    def rem_template(rem):
        """
        Provides the text for the rem section of a Q-Chem input file.

        Args:
            rem (dict): Rem section data to be written.
        Return:
            str
        """
        rem_list = list()
        rem_list.append("$rem")
        for key, value in rem.items():
            rem_list.append("   {key} = {value}".format(key=key, value=value))
        rem_list.append("$end")
        return '\n'.join(rem_list)

    @staticmethod
    def opt_template(opt):
        """
        Provides the text for the opt section of a Q-Chem input file.

        Args:
            opt (dict of dicts): Opt section data to be written.
        Return:
            str
        """
        opt_list = list()
        opt_list.append("$opt")
        # loops over all opt sections
        for key, value in opt.items():
            opt_list.append("{section}".format(section=key))
            # loops over all values within the section
            for i in value:
                opt_list.append("   {val}".format(val=i))
            opt_list.append("END{section}".format(section=key))
            opt_list.append("")
        # this deletes the empty space after the last section
        del opt_list[-1]
        opt_list.append("$end")
        return '\n'.join(opt_list)

    @staticmethod
    def pcm_template(pcm):
        """
        Provides the text for the pcm section of a Q-Chem input file.

        Args:
            molecule (dict): PCM section data to be written.
        Return:
            str
        """
        pcm_list = list()
        pcm_list.append("$pcm")
        for key, value in pcm.items():
            pcm_list.append("   {key} {value}".format(key=key, value=value))
        pcm_list.append("$end")
        return '\n'.join(pcm_list)

    @staticmethod
    def solvent_template(solvent):
        """
        Provides the text for the solvent section of a Q-Chem input file.

        Args:
            solvent (dict): Solvent section data to be written.
        Return:
            str
        """
        solvent_list = list()
        solvent_list.append("$solvent")
        for key, value in solvent.items():
            solvent_list.append("   {key} {value}".format(
                key=key, value=value))
        solvent_list.append("$end")
        return '\n'.join(solvent_list)

    @staticmethod
    def smx_template(smx):
        """
        Provides the text for the smx section of a Q-Chem input file.

        Args:
            smx (dict): SMX section data to be written.
        Return:
            str
        """
        smx_list = list()
        smx_list.append("$smx")
        for key, value in smx.items():
            if value == "tetrahydrofuran":
                smx_list.append("   {key} {value}".format(
                    key=key, value="thf"))
            else:
                smx_list.append("   {key} {value}".format(
                    key=key, value=value))
        smx_list.append("$end")
        return '\n'.join(smx_list)

    @staticmethod
    def scan_template(scan):
        """
        Provides the text for the scan section of a Q-Chem input file.

        Args:
            scan (dict of dicts): Scan section data to be written.
        Return:
            str
        """
        scan_list = list()
        scan_list.append("$scan")
        total_vars = sum([len(v) for v in scan.values()])
        if total_vars > 2:
            raise ValueError("Q-Chem only supports PES_SCAN with two or less "
                             "variables.")
        for var_type, variables in scan.items():
            if variables not in [None, list()]:
                for var in variables:
                    scan_list.append("   {var_type} {var}".format(
                        var_type=var_type, var=var))
        scan_list.append("$end")
        return '\n'.join(scan_list)

    @staticmethod
    def plots_template(plots):
        """
        Provides the text for the plots section of a Q-Chem input file.

        Args:
            plots (dict): Plots section data to be written.
        Return:
            str
        """
        plots_list = list()
        plots_list.append("$plots")
        for key, value in plots.items():
            plots_list.append("   {key} {value}".format(
                key=key, value=value))
        plots_list.append("$end")
        return '\n'.join(plots_list)

    @staticmethod
    def find_sections(string):
        """
        Determine what input sections (rem, opt, molecule, smx, etc.) are
            present in a string.

        Args:
            string (str): String representing a Q-Chem input file.
        Returns:
            sections (list of str): The list of input file sections present in
                the given text.
        """
        patterns = {"sections": r"^\s*?\$([a-z]+)", "multiple_jobs": r"(@@@)"}
        matches = read_pattern(string, patterns)
        # list of the sections present
        sections = [val[0] for val in matches["sections"] if val[0] != "end"]

        # TODO: this error should be replaced by a multi job read function when it is added
        if "multiple_jobs" in matches.keys():
            raise ValueError(
                "Input file contains multiple Q-Chem jobs please parse separately"
            )

        if "molecule" not in sections:
            raise ValueError("Input file does not contain a molecule section")
        if "rem" not in sections:
            raise ValueError("Input file does not contain a rem section")
        return sections

    @classmethod
    def read_molecule(cls, string):
        """
        Parse a Q-Chem input file's molecule section and extract the molecule.

        Args:
            string: Text of a Q-Chem input file.
        Returns:
            Molecule or str
        """

        # TODO: What happens if there is no molecule section?
        # This extends to any section - need to add checks.

        charge = None
        spin_mult = None
        patterns = {
            "read": r"^\s*\$molecule\n\s*(read)",
            "charge": r"^\s*\$molecule\n\s*((?:\-)*\d+)\s+\d",
            "spin_mult": r"^\s*\$molecule\n\s(?:\-)*\d+\s*(\d)"
        }
        matches = read_pattern(string, patterns)

        if "read" in matches.keys():
            return "read"
        if "charge" in matches.keys():
            charge = float(matches["charge"][0][0])
        if "spin_mult" in matches.keys():
            spin_mult = int(matches["spin_mult"][0][0])

        header = r"^\s*\$molecule\n\s*(?:\-)*\d+\s*\d"
        row = r"\s*((?i)[a-z]+)\s+([\d\-\.]+)\s+([\d\-\.]+)\s+([\d\-\.]+)"
        footer = r"^\$end"

        mol_table = read_table_pattern(
            string,
            header_pattern=header,
            row_pattern=row,
            footer_pattern=footer)

        species = [val[0] for val in mol_table[0]]
        coords = [[float(val[1]), float(val[2]),
                   float(val[3])] for val in mol_table[0]]

        return Molecule(species=species, coords=coords, charge=charge,
                        spin_multiplicity=spin_mult)

    @staticmethod
    def read_rem(string):
        """
        Parse a Q-Chem input file's rem section and extract the relevant data.

        Args:
            string: Text of a Q-Chem input file.
        Returns:
            dict
        """

        header = r"^\s*\$rem"
        row = r"\s*([a-zA-Z\_]+)\s*=?\s*(\S+)"
        footer = r"^\s*\$end"
        rem_table = read_table_pattern(
            string,
            header_pattern=header,
            row_pattern=row,
            footer_pattern=footer)

        return {key: val for key, val in rem_table[0]}

    @staticmethod
    def read_opt(string):
        """
        Parse a Q-Chem input file's opt section and extract the relevant data.

        Args:
            string: Text of a Q-Chem input file.
        Returns:
            opt (dict of dicts): Opt section data.
        """

        patterns = {
            "CONSTRAINT": r"^\s*CONSTRAINT",
            "FIXED": r"^\s*FIXED",
            "DUMMY": r"^\s*DUMMY",
            "CONNECT": r"^\s*CONNECT"
        }
        opt_matches = read_pattern(string, patterns)
        opt_sections = [key for key in opt_matches.keys()]

        opt = dict()
        if "CONSTRAINT" in opt_sections:
            c_header = r"^\s*CONSTRAINT\n"
            c_row = r"(\w.*)\n"
            c_footer = r"^\s*ENDCONSTRAINT\n"
            c_table = read_table_pattern(
                string,
                header_pattern=c_header,
                row_pattern=c_row,
                footer_pattern=c_footer)
            opt["CONSTRAINT"] = [val[0] for val in c_table[0]]
        if "FIXED" in opt_sections:
            f_header = r"^\s*FIXED\n"
            f_row = r"(\w.*)\n"
            f_footer = r"^\s*ENDFIXED\n"
            f_table = read_table_pattern(
                string,
                header_pattern=f_header,
                row_pattern=f_row,
                footer_pattern=f_footer)
            opt["FIXED"] = [val[0] for val in f_table[0]]
        if "DUMMY" in opt_sections:
            d_header = r"^\s*DUMMY\n"
            d_row = r"(\w.*)\n"
            d_footer = r"^\s*ENDDUMMY\n"
            d_table = read_table_pattern(
                string,
                header_pattern=d_header,
                row_pattern=d_row,
                footer_pattern=d_footer)
            opt["DUMMY"] = [val[0] for val in d_table[0]]
        if "CONNECT" in opt_sections:
            cc_header = r"^\s*CONNECT\n"
            cc_row = r"(\w.*)\n"
            cc_footer = r"^\s*ENDCONNECT\n"
            cc_table = read_table_pattern(
                string,
                header_pattern=cc_header,
                row_pattern=cc_row,
                footer_pattern=cc_footer)
            opt["CONNECT"] = [val[0] for val in cc_table[0]]

        return opt

    @staticmethod
    def read_pcm(string):
        """
        Parse a Q-Chem input file's pcm section and extract the relevant data.

        Args:
            string: Text of a Q-Chem input file.
        Returns:
            dict
        """

        header = r"^\s*\$pcm"
        row = r"\s*([a-zA-Z\_]+)\s+(\S+)"
        footer = r"^\s*\$end"

        pcm_table = read_table_pattern(
            string,
            header_pattern=header,
            row_pattern=row,
            footer_pattern=footer)

        if len(pcm_table) == 0:
            print(
                "No valid PCM inputs found. Note that there should be no '=' "
                "characters in PCM input lines."
            )
            return dict()
        else:
            return {key: val for key, val in pcm_table[0]}

    @staticmethod
    def read_solvent(string):
        """
        Parse a Q-Chem input file's solvent section and extract the relevant
            data.

        Args:
            string: Text of a Q-Chem input file.
        Returns:
            dict
        """

        header = r"^\s*\$solvent"
        row = r"\s*([a-zA-Z\_]+)\s+(\S+)"
        footer = r"^\s*\$end"

        solvent_table = read_table_pattern(
            string,
            header_pattern=header,
            row_pattern=row,
            footer_pattern=footer)

        if len(solvent_table) == 0:
            print(
                "No valid solvent inputs found. Note that there should be no "
                "'=' characters in solvent input lines."
            )
            return dict()
        else:
            return {key: val for key, val in solvent_table[0]}

    @staticmethod
    def read_smx(string):
        """
        Parse a Q-Chem input file's smx section and extract the relevant data.

        Args:
            string: Text of a Q-Chem input file.
        Returns:
            smx (dict): SMx section data.
        """

        header = r"^\s*\$smx"
        row = r"\s*([a-zA-Z\_]+)\s+(\S+)"
        footer = r"^\s*\$end"

        smx_table = read_table_pattern(
            string,
            header_pattern=header,
            row_pattern=row,
            footer_pattern=footer)

        if len(smx_table) == 0:
            print(
                "No valid smx inputs found. Note that there should be no '=' "
                "characters in smx input lines."
            )
            return dict()
        else:
            smx = {key: val for key, val in smx_table[0]}
            if smx["solvent"] == "tetrahydrofuran":
                smx["solvent"] = "thf"
            return smx

    @staticmethod
    def read_scan(string):
        """
        Parse a Q-Chem input file's scan section and extract the relevant data.

        Args:
            string: Text of a Q-Chem input file.
        Returns:
            dict
        """

        header = r"^\s*\$scan"
        row = r"\s*(stre|bend|tors|STRE|BEND|TORS)\s+((?:[\-\.0-9]+\s*)+)"
        footer = r"^\s*\$end"

        scan_table = read_table_pattern(string,
                                        header_pattern=header,
                                        row_pattern=row,
                                        footer_pattern=footer)

        if scan_table == list():
            print(
                "No valid scan inputs found. Note that there should be no '=' "
                "characters in scan input lines."
            )
            return dict()
        else:
            stre = list()
            bend = list()
            tors = list()
            for row in scan_table[0]:
                if row[0].lower() == "stre":
                    stre.append(row[1].replace("\n", "").rstrip())
                elif row[0].lower() == "bend":
                    bend.append(row[1].replace("\n", "").rstrip())
                elif row[0].lower() == "tors":
                    tors.append(row[1].replace("\n", "").rstrip())

            if len(stre) + len(bend) + len(tors) > 2:
                raise ValueError("No more than two variables are allows in the "
                                 "scan section!")

            return {"stre": stre, "bend": bend, "tors": tors}

    @staticmethod
    def read_plots(string):
        """
        Parse a Q-Chem input file's plots section and extract the relevant data.

        Args:
            string: Text of a Q-Chem input file.
        Returns:
            dict
        """

        header = r"^\s*\$plots"
        row = r"\s*([a-zA-Z\_]+)\s+(\S+)"
        footer = r"^\s*\$end"

        plots_table = read_table_pattern(
            string,
            header_pattern=header,
            row_pattern=row,
            footer_pattern=footer)

        if len(plots_table) == 0:
            print(
                "No valid plots inputs found. Note that there should be no '=' characters in plots input lines."
            )
            return dict()

        else:
            return {key: val for key, val in plots_table[0]}
