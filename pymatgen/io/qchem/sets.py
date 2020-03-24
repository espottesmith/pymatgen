# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import logging
import os
from monty.io import zopen
from pymatgen.io.qchem.inputs import QCInput
from pymatgen.io.qchem.utils import lower_and_check_unique

# Classes defining defaults for various kinds of Q-Chem jobs.

__author__ = "Samuel Blau, Brandon Wood, Shyam Dwaraknath, Evan Spotte-Smith"
__copyright__ = "Copyright 2018, The Materials Project"
__version__ = "0.1"

logger = logging.getLogger(__name__)


class QChemDictSet(QCInput):
    """
    Build a QCInput given all the various input parameters. Can be extended by
        standard implementations in pymatgen.io.qchem.sets.
    """

    def __init__(self,
                 molecule,
                 job_type,
                 basis_set,
                 scf_algorithm,
                 dft_rung=4,
                 pcm_dielectric=None,
                 smd_solvent=None,
                 custom_smd=None,
                 scan_variables=None,
                 opt_variables=None,
                 max_scf_cycles=200,
                 geom_opt_max_cycles=200,
                 plot_cubes=False,
                 overwrite_inputs=None):
        """
        Args:
            molecule (Molecule or str): Input molecule. Can be "read", meaning
                that input will be read in from the output of a previous
                calculation.
            job_type (str): Type of calculation to be conducted. At present,
                pymatgen supports values in ["opt", "optimization", "freq",
                "frequency", "sp", "force", "pes_scan", "ts", "nmr"]
            basis_set (str): Q-Chem basis set (e.g. "6-311++g(d,p)"). Note that
                pymatgen does not check that the basis set provided is allowed
                by Q-Chem.
            scf_algorithm (str): Algorithm to be used to converge
                self-consistent field (SCF) calculations. One of ["diis", "dm",
                "diis_dm", "diis_gdm", "gdm", "rca", "rcs_diis", "roothan"].
                Note that pymatgen does not check that the algorithm provided
                is allowed by Q-Chem.
            dft_rung (int): Integer between 1 and 5 defining the "rung" on a
                custom "Jacob's Ladder" of functionals. Higher rungs use more
                expensive - but also more accurate - functionals. For instance,
                dft_rung = 1 => B3LYP; dft_rung = 5 => wb97mv
            pcm_dielectric (float): Value of the dielectric constant for a
                solvent to be used with the PCM implicit solvent model. Note
                that, if pcm_dielectric is not None, then smd_solvent should be
                None.
            smd_solvent (str): Solvent to be used with the SMD implicit solvent
                method. If a custom solvent is to be used use
                smd_solvent="custom" or smd_solvent="other".
            custom_smd (str): If a non-standard solvent is to be used, the SMD
                parameters should be put in a comma-separated string. The
                parameters are, in order:
                - dielectric constant
                - refractive index
                - bulk surface tension
                - Abraham's acidity
                - Abraham's basicity
                - Carbon aromaticity
                - Electronegative halogenicity
                Ex: for ethylene carbonate,
                custom_smd="18.5,1.415,0.00,0.735,20.2,0.00,0.00"
            scan_variables (dict of lists): If job_type="pes_scan", then
                coordinates to be scanned over should be included here. Because
                two constraints of the same type are allowed (for instance, two
                torsions or two bond stretches), each TYPE of variable (stre,
                bend, tors) should be its own key in the dict, rather than each
                variable. Note that the total number of variable (sum of
                lengths of all lists) CANNOT be more than two.
                Ex: scan_variables={"stre": ["3 6 1.5 1.9 0.1"],
                                    "tors": ["1 2 3 4 -180 180 15"]}
            opt_variables (dict): For constrained optimization-type jobs. Each
                opt section is a key and the corresponding values are a list of
                strings. Stings must be formatted as instructed by the Q-Chem
                manual. The different opt sections are: CONSTRAINT, FIXED,
                DUMMY, and CONNECT
                Ex: opt_variables={"CONSTRAINT": ["tors 2 3 4 5 25.0",
                                                  "tors 2 5 7 9 80.0"],
                                   "FIXED": ["2 XY"]}
            max_scf_cycles (int): For all self-consistent field (SCF)
                calculations, allow this many iterations to converge before the
                program should terminate. Values between 200-300 are generally
                appropriate.
            geom_opt_max_cycles (int): For all jobs involving optimization (opt,
                ts, pes_scan, etc.), allow this many iterations in the geometry
                optimization algorithm before the terminate should terminate.
                Values between 100-200 are generally appropriate, though
                large/difficult structures, or structures with poor initial
                guesses, may require more iterations.
            plot_cubes (bool): If True (default False), then Q-Chem will plot
                the molecular orbitals of the molecule.
            overwrite_inputs (dict): This is dictionary of QChem input sections
            to add or overwrite variables, the available sections are currently
            rem, pcm, and solvent. So the accepted keys are rem, pcm, or solvent
            and the value is a dictionary of key value pairs relevant to the
            section. An example would be adding a new variable to the rem
            section that sets symmetry to false.
            ex. overwrite_inputs = {"rem": {"symmetry": "false"}}
            ***NOTE: overwrite_inputs supercedes all defaults.***
        """

        self.molecule = molecule
        self.job_type = job_type
        self.basis_set = basis_set
        self.scf_algorithm = scf_algorithm
        self.dft_rung = dft_rung
        self.pcm_dielectric = pcm_dielectric
        self.smd_solvent = smd_solvent
        self.custom_smd = custom_smd
        self.scan_variables = scan_variables
        self.opt_variables = opt_variables
        self.max_scf_cycles = max_scf_cycles
        self.geom_opt_max_cycles = geom_opt_max_cycles
        self.plot_cubes = plot_cubes
        self.overwrite_inputs = overwrite_inputs

        myrem = dict()
        mypcm = dict()
        mysolvent = dict()
        mysmx = dict()
        myopt = dict()
        myscan = dict()
        myplots = dict()

        # Fill opt and scan sections based on user input
        if self.opt_variables is not None:
            myopt = self.opt_variables

        if self.scan_variables is not None:
            myscan = self.scan_variables

        # Populate rem section based on defaults
        myrem["job_type"] = job_type
        myrem["basis"] = self.basis_set
        myrem["max_scf_cycles"] = self.max_scf_cycles
        myrem["gen_scfman"] = "true"
        myrem["xc_grid"] = "3"
        myrem["scf_algorithm"] = self.scf_algorithm
        myrem["resp_charges"] = "true"
        myrem["symmetry"] = "false"
        myrem["sym_ignore"] = "true"

        # Custom "Jacob's Ladder"
        if self.dft_rung == 1:
            myrem["method"] = "b3lyp"
        elif self.dft_rung == 2:
            myrem["method"] = "b3lyp"
            myrem["dft_D"] = "D3_BJ"
        elif self.dft_rung == 3:
            myrem["method"] = "wb97xd"
        elif self.dft_rung == 4:
            myrem["method"] = "wb97xv"
        elif self.dft_rung == 5:
            myrem["method"] = "wb97mv"
        else:
            raise ValueError("dft_rung should be between 1 and 5!")

        # Only jobs that require optimization should limit the optimization
        # cycles
        if self.job_type.lower() in ["opt", "optimization", "ts", "pes_scan"]:
            myrem["geom_opt_max_cycles"] = self.geom_opt_max_cycles

        # Control implicit solvation methods based on user-provided solvent
        # information
        if self.pcm_dielectric is not None and self.smd_solvent is not None:
            raise ValueError("Only one of pcm or smd may be used for solvation.")

        if self.pcm_dielectric is not None:
            mypcm = {"heavypoints": "194",
                     "hpoints": "194",
                     "radii": "uff",
                     "theory": "cpcm",
                     "vdwscale": "1.1"}

            mysolvent["dielectric"] = self.pcm_dielectric
            myrem["solvent_method"] = 'pcm'

        if self.smd_solvent is not None:
            if self.smd_solvent == "custom":
                mysmx["solvent"] = "other"
            else:
                mysmx["solvent"] = self.smd_solvent
            myrem["solvent_method"] = "smd"
            myrem["ideriv"] = "1"
            if self.smd_solvent == "custom" or self.smd_solvent == "other":
                if self.custom_smd is None:
                    raise ValueError('A user-defined SMD requires passing '
                                     'custom_smd, a string of seven'
                                     'comma-separated values in the following'
                                     'order: dielectric, refractive index, '
                                     'acidity, basicity, surface tension,'
                                     'aromaticity, electronegative '
                                     'halogenicity')

        # Control electron density plotting based on default values
        if self.plot_cubes:
            myplots = {"grid_spacing": "0.05",
                       "total_density": "0"}
            myrem["plots"] = "true"
            myrem["make_cube_files"] = "true"

        # Populate input sections based on overwritten values
        if self.overwrite_inputs:
            for sec, sec_dict in self.overwrite_inputs.items():
                if sec == "rem":
                    temp_rem = lower_and_check_unique(sec_dict)
                    for k, v in temp_rem.items():
                        myrem[k] = v
                if sec == "opt":
                    temp_opt = lower_and_check_unique(sec_dict)
                    for k, v in temp_opt.items():
                        myopt[k] = v
                if sec == "pcm":
                    temp_pcm = lower_and_check_unique(sec_dict)
                    for k, v in temp_pcm.items():
                        mypcm[k] = v
                if sec == "solvent":
                    temp_solvent = lower_and_check_unique(sec_dict)
                    for k, v in temp_solvent.items():
                        mysolvent[k] = v
                if sec == "smx":
                    temp_smx = lower_and_check_unique(sec_dict)
                    for k, v in temp_smx.items():
                        mysmx[k] = v
                if sec == "scan":
                    temp_scan = lower_and_check_unique(sec_dict)
                    for k, v in temp_scan.items():
                        myscan[k] = v
                if sec == "plots":
                    tmp_plots = lower_and_check_unique(sec_dict)
                    for k, v in tmp_plots.items():
                        myplots[k] = v

        super().__init__(self.molecule, rem=myrem, opt=myopt, pcm=mypcm,
                         solvent=mysolvent, smx=mysmx, scan=myscan,
                         plots=myplots)

    def write(self, input_file):
        """
        Write input file, as well as any auxiliary files necessary for the job.

        Args:
            input_file (str): File name or path to the Q-Chem input file.
        Returns:
            None
        """

        self.write_file(input_file)
        # SMX methods in Q-Chem require a "solvent_data" file to be written
        # if a custom solvent is to be used
        if self.smd_solvent == "custom" or self.smd_solvent == "other":
            with zopen(os.path.join(os.path.dirname(input_file), "solvent_data"), 'wt') as f:
                f.write(self.custom_smd)


class OptSet(QChemDictSet):
    """
    QChemDictSet for a geometry optimization
    """

    def __init__(self,
                 molecule,
                 dft_rung=3,
                 basis_set="def2-tzvppd",
                 pcm_dielectric=None,
                 smd_solvent=None,
                 custom_smd=None,
                 scf_algorithm="diis",
                 max_scf_cycles=200,
                 geom_opt_max_cycles=200,
                 plot_cubes=False,
                 overwrite_inputs=None):

        self.basis_set = basis_set
        self.scf_algorithm = scf_algorithm
        self.max_scf_cycles = max_scf_cycles
        self.geom_opt_max_cycles = geom_opt_max_cycles

        super().__init__(
            molecule=molecule,
            job_type="opt",
            dft_rung=dft_rung,
            pcm_dielectric=pcm_dielectric,
            smd_solvent=smd_solvent,
            custom_smd=custom_smd,
            basis_set=self.basis_set,
            scf_algorithm=self.scf_algorithm,
            max_scf_cycles=self.max_scf_cycles,
            geom_opt_max_cycles=self.geom_opt_max_cycles,
            plot_cubes=plot_cubes,
            overwrite_inputs=overwrite_inputs)


class TransitionStateSet(QChemDictSet):
    """
    QChemDictSet for a transition-state geometry optimization
    """

    def __init__(self,
                 molecule,
                 dft_rung=3,
                 basis_set="def2-tzvppd",
                 pcm_dielectric=None,
                 smd_solvent=None,
                 custom_smd=None,
                 scf_algorithm="diis",
                 max_scf_cycles=200,
                 geom_opt_max_cycles=200,
                 plot_cubes=False,
                 overwrite_inputs=None):

        self.basis_set = basis_set
        self.scf_algorithm = scf_algorithm
        self.max_scf_cycles = max_scf_cycles
        self.geom_opt_max_cycles = geom_opt_max_cycles

        super(TransitionStateSet, self).__init__(
            molecule=molecule,
            job_type="ts",
            dft_rung=dft_rung,
            pcm_dielectric=pcm_dielectric,
            smd_solvent=smd_solvent,
            custom_smd=custom_smd,
            basis_set=self.basis_set,
            scf_algorithm=self.scf_algorithm,
            max_scf_cycles=self.max_scf_cycles,
            geom_opt_max_cycles=self.geom_opt_max_cycles,
            plot_cubes=plot_cubes,
            overwrite_inputs=overwrite_inputs)


class SinglePointSet(QChemDictSet):
    """
    QChemDictSet for a single point calculation
    """

    def __init__(self,
                 molecule,
                 dft_rung=3,
                 basis_set="def2-tzvppd",
                 pcm_dielectric=None,
                 smd_solvent=None,
                 custom_smd=None,
                 scf_algorithm="diis",
                 max_scf_cycles=200,
                 plot_cubes=False,
                 overwrite_inputs=None):

        self.basis_set = basis_set
        self.scf_algorithm = scf_algorithm
        self.max_scf_cycles = max_scf_cycles

        super().__init__(
            molecule=molecule,
            job_type="sp",
            dft_rung=dft_rung,
            pcm_dielectric=pcm_dielectric,
            smd_solvent=smd_solvent,
            custom_smd=custom_smd,
            basis_set=self.basis_set,
            scf_algorithm=self.scf_algorithm,
            max_scf_cycles=self.max_scf_cycles,
            plot_cubes=plot_cubes,
            overwrite_inputs=overwrite_inputs)


class ForceSet(QChemDictSet):
    """
    QChemDictSet for a force (gradient) calculation
    """

    def __init__(self,
                 molecule,
                 dft_rung=3,
                 basis_set="def2-tzvppd",
                 pcm_dielectric=None,
                 smd_solvent=None,
                 custom_smd=None,
                 scf_algorithm="diis",
                 max_scf_cycles=200,
                 overwrite_inputs=None):

        self.basis_set = basis_set
        self.scf_algorithm = scf_algorithm
        self.max_scf_cycles = max_scf_cycles

        super().__init__(
            molecule=molecule,
            job_type="force",
            dft_rung=dft_rung,
            pcm_dielectric=pcm_dielectric,
            smd_solvent=smd_solvent,
            custom_smd=custom_smd,
            basis_set=self.basis_set,
            scf_algorithm=self.scf_algorithm,
            max_scf_cycles=self.max_scf_cycles,
            overwrite_inputs=overwrite_inputs)


class FreqSet(QChemDictSet):
    """
    QChemDictSet for a frequency calculation
    """

    def __init__(self,
                 molecule,
                 dft_rung=3,
                 basis_set="def2-tzvppd",
                 pcm_dielectric=None,
                 smd_solvent=None,
                 custom_smd=None,
                 scf_algorithm="diis",
                 max_scf_cycles=200,
                 plot_cubes=False,
                 overwrite_inputs=None):

        self.basis_set = basis_set
        self.scf_algorithm = scf_algorithm
        self.max_scf_cycles = max_scf_cycles

        super().__init__(
            molecule=molecule,
            job_type="freq",
            dft_rung=dft_rung,
            pcm_dielectric=pcm_dielectric,
            smd_solvent=smd_solvent,
            custom_smd=custom_smd,
            basis_set=self.basis_set,
            scf_algorithm=self.scf_algorithm,
            max_scf_cycles=self.max_scf_cycles,
            plot_cubes=plot_cubes,
            overwrite_inputs=overwrite_inputs)


class PESScanSet(QChemDictSet):
    """
    QChemDictSet for a potential energy surface scan (PES_SCAN) calculation,
    used primarily to identify possible transition states or to sample different
    geometries.

    Note: Because there are no defaults that can be used for a PES scan (the
    variables are completely dependent on the molecular structure), by default
    scan_variables = None. However, a PES Scan job should not be run with less
    than one variable (or more than two variables).
    """

    def __init__(self,
                 molecule,
                 dft_rung=3,
                 basis_set="def2-tzvppd",
                 pcm_dielectric=None,
                 smd_solvent=None,
                 custom_smd=None,
                 scan_variables=None,
                 scf_algorithm="diis",
                 max_scf_cycles=200,
                 overwrite_inputs=None):

        self.basis_set = basis_set
        self.scf_algorithm = scf_algorithm
        self.max_scf_cycles = max_scf_cycles

        if scan_variables is None:
            raise ValueError("Cannot run a pes_scan job without some variable "
                             "to scan over!")

        super(PESScanSet, self).__init__(
            molecule=molecule,
            job_type="pes_scan",
            dft_rung=dft_rung,
            pcm_dielectric=pcm_dielectric,
            smd_solvent=smd_solvent,
            custom_smd=custom_smd,
            scan_variables=scan_variables,
            basis_set=self.basis_set,
            scf_algorithm=self.scf_algorithm,
            max_scf_cycles=self.max_scf_cycles,
            overwrite_inputs=overwrite_inputs)
