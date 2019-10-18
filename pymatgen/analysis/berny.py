import copy
from pymatgen.core.structure import Molecule, Structure

from berny import Berny, geomlib


class BernyOptimizer:

    """
    This class wraps the optimizer in berny, which is based closely on the
    methods described in Birkholz, A.B. and Schlegel, H.B., 2016. Theoretical
    Chemistry Accounts, 135(4), p.84.

    The Berny optimizer uses energy and gradient calculations from any program
    (MOPAC, Q-Chem, Gaussian, VASP) to generate guess structures for relaxed
    structures (minima) and transition states (saddle points).

    Note: Most (but not all) default values are based on the default values
    given in berny.

    Args:
        chemistry (Molecule or Structure): The chemical to be optimized
        prev_calc_data (dict): Start an optimization from a previous state,
            output from berny
        logger (object): The logger can be any object class with a "log" method.
            By default, this is None, meaning that the default pyberny logger
            will be used
        verbosity (int): If the default berny logger is to be used, this
            variable can set the level of output. If verbosity is not given,
            the value will be set to 0 (maximum verbosity)
        transition_state (bool): If True (default False), search for a saddle
            point of the potential energy surface, rather than a minimum.
        max_steps (int): If the optimization has not converged after this many
            steps, the job will be considered to have failed.
        max_gradient (float): Convergence parameter
        rms_gradient (float): Convergence parameter
        max_step_size (float): Convergence parameter
        rms_step_size (float): Convergence parameter
        trust (float): This sets the initial trust radius (region where
            steps based on quadratic approximations are considered safe). Any
            suggested step that moves outside of the trust radius will be
            adjusted
        dihedral (bool): If True (default), consider dihedral angles in the
            construction of internal coordinates
        weak_dihedral (bool: If True (default False), consider dihedral angles
            comprised of two or more noncovalent bonds


    """

    def __init__(self, chemistry, prev_calc_data=None, logger=None,
                 verbosity=None, transition_state=False, max_steps=250,
                 max_gradient=4.5e-4, rms_gradient=3.0e-4, max_step_size=1.8e-3,
                 rms_step_size=1.2e-3, trust=0.3, dihedral=True, weak_dihedral=False):

        self.initial_chemistry = chemistry
        self.chemistry = self.initial_chemistry

        if isinstance(self.chemistry, Structure):
            self.lattice = self.chemistry.lattice.matrix

        elif isinstance(self.chemistry, Molecule):
            self.lattice = None
        else:
            raise ValueError("chemistry must be either a Structure or Molecule "
                             "object!")

        geom = geomlib.Geometry([str(e.species) for e in self.chemistry],
                                [e.coords for e in self.chemistry],
                                lattice=self.lattice)

        self.logger = logger
        self.transition_state = transition_state

        self.params = {"gradientmax": max_gradient,
                       "gradientrms": rms_gradient,
                       "stepmax": max_step_size,
                       "steprms": rms_step_size,
                       "trust": trust,
                       "dihedral": dihedral,
                       "superweakdih": weak_dihedral}

        if logger is None and verbosity is not None:
            verbosity = None

        self.berny = Berny(geom, self.logger, debug=True,
                           restart=prev_calc_data, maxsteps=max_steps,
                           verbosity=verbosity,
                           transition_state=self.transition_state,
                           params=self.params)

        self.state = dict(self.berny.state)

    def update(self, energy, gradients):
        """
        Update the state of the optimizer based on current energy and gradients.

        Args:
        energy (float): The energy at the current geometry
        gradients (list of floats): An Nx3 list of floats representing
            the gradients in Cartesian values

        Returns:
            None
        """

        self.state = self.berny.send((energy, gradients))

    def get_next_geometry(self):
        """
        Access the next value of the berny optimizer generator

        Args:
            None

        Returns:
            (new_chemistry, converged): either a Lattice or Molecule, depending on the type
                of the type of self.chemistry
        """

        # Returns the geometry of the current state in Cartesian coordinates
        new_geom = next(self.berny, None)

        if new_geom is None and not self.berny.converged:
            return None, False

        if self.berny.converged:
            new_geom = self.state.geom

        if isinstance(self.initial_chemistry, Structure):
            return (Structure(lattice=self.lattice,
                              species=new_geom.species,
                              coords=new_geom.coords,
                              charge=self.initial_chemistry.charge),
                    self.berny.converged)
        elif isinstance(self.initial_chemistry, Molecule):
            return (Molecule(species=new_geom.species,
                             coords=new_geom.coords,
                             charge=self.initial_chemistry.charge,
                             spin_multiplicity=self.initial_chemistry.spin_multiplicity),
                    self.berny.converged)