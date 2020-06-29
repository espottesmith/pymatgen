import os
from typing import List, Dict, Optional
import copy

from schrodinger.application.jaguar.reactiq_input import (ReactiqInput)

from monty.json import MSONable

from pymatgen.core.structure import Molecule
from pymatgen.io.schrodinger.schrodinger_adapter import (molecule_to_maestro_file,
                                                         maestro_file_to_molecule)
from pymatgen.io.schrodinger.autots_input import AutoTSInput


class AutoTSSet(AutoTSInput):
    """
    Build an AutoTSInput given various input parameters.
    """

    def __init__(self,
                 reactants,
                 products,
                 basis_set="def2-tzvpd",
                 dft_rung=4,
                 pcm_dielectric=None,
                 max_scf_cycles=400,
                 geom_opt_max_cycles=250,
                 overwrite_inputs_autots=None,
                 overwrite_inputs_gen=None):
        """
        Args:
            reactants (list of Molecule objects):
            products (list of Molecule objects):
            basis_set (str):
            dft_rung (int):
            pcm_dielectric (float):
            max_scf_cycles (int):
            geom_opt_max_cycles (int):
            overwrite_inputs_autots (dict): Dictionary to overwrite default
                AutoTS-specific parameters
            overwrite_inputs_gen (dict): Dictionary to overwrite default
                Jaguar gen parameters
        """

        autots_variables = {"eliminate_multiple_frequencies": True,
                            "free_energy": True,
                            "require_irc_success": True,
                            "ts_vet_max_freq": -40.0,
                            "units": "ev",
                            "use_alternate": True}

        if dft_rung == 1:
            dftname = "b3lyp"
        elif dft_rung == 2:
            dftname = "cam-b3lyp"
        elif dft_rung == 3:
            dftname = "cam-b3lyp-d3"
        elif dft_rung == 4:
            dftname = "wb97x-d"
        else:
            raise ValueError("Invalid dft_rung provided!")

        gen_variables = {"dftname": dftname,
                         "basis": basis_set,
                         "babel": "xyz",
                         "ip472": 2,  # Output all steps of geometry optimization in *.mae
                         "ip172": 2,  # Print RESP file
                         "ip175": 2,  # Print XYZ files
                         "ifreq": 1,  # Frequency calculation
                         "irder": 1,  # IR vibrational modes calculated
                         "nmder": 2,  # Numerical second derivatives
                         "nogas": 2,  # Skip gas-phase optimization, if PCM is used
                         "maxitg": geom_opt_max_cycles,  # Maximum number of geometry optimization iterations
                         "intopt_switch": 0,  # Do not switch from internal to Cartesian coordinates
                         "optcoord_update": 0,  # Do not run checks to change coordinate system
                         "props_each_step": 1,  # Calculate properties at each optimization step
                         # "iaccg": 5  # Tight convergence criteria for optimization
                         "mulken": 1,  # Calculate Mulliken properties by atom
                         "maxit": max_scf_cycles,  # Maximum number of SCF iterations
                         "iacc": 2,  # Use "accurate" SCF convergence criteria
                         # "noauto": 3  # All calculations done on fine grid
                         "isymm": 0,  # Do not use symmetry
                         "espunit": 6  # Electrostatic potential in units of eV
                         }

        if pcm_dielectric is not None:
            gen_variables["isolv"] = 7
            gen_variables["epsout"] = pcm_dielectric
            gen_variables["pcm_model"] = "cosmo"

        if overwrite_inputs_autots is not None:
            for key, value in overwrite_inputs_autots.items():
                autots_variables[key] = value

        if overwrite_inputs_gen is not None:
            for key, value in overwrite_inputs_gen.items():
                gen_variables[key] = value

        super().__init__(reactants, products, autots_variables, gen_variables)