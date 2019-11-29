from pymatgen.core.structure import Molecule, Structure

from berny import Berny, State, geomlib


class BernyLogger:

    """
    This class serves as a substitute for the pyberny Logger class. It
    prints all output from optimization to an output stream (by default, a
    file).

    Args:
        verbosity (int): Verbosity level (the higher the level, the more
            output the user will receive). Default is None, meaning the user
            will receive all output.
        logfile (str): File or output stream where all Berny output will
            go. By default, this is "berny.log", but it could be some other
            file or a stream like sys.stdout or sys.stderr
    """

    def __init__(self, verbosity=None, logfile="berny.log"):
        self.verbosity = verbosity
        self.logfile = logfile

    def __call__(self, msg, level=0):
        """
        Log a message

        Args:
            msg:
            level:

        Returns: None
        """

        msgstr = str(msg)
        if not msgstr.endswith("\n"):
            msgstr += "\n"

        if self.verbosity is None:
            with open(self.logfile, "a+") as log:
                log.write(msgstr)
        elif level < self.verbosity:
            with open(self.logfile, "a+") as log:
                log.write(msgstr)


class BernyOptimizer:

    """
    This class wraps the optimizer in berny, which is based closely on the
    methods described in Birkholz, A.B. and Schlegel, H.B., 2016. Theoretical
    Chemistry Accounts, 135(4), p.84.

    The Berny optimizer uses energy and gradient calculations from any program
    (MOPAC, Q-Chem, Gaussian) to generate guess structures for relaxed
    structures (minima) and transition states (saddle points). It is presently
    designed only for use with molecules, though in principle it could be used
    with periodic structures as well.

    Note: Most (but not all) default values are based on the default values
    given in berny.

    Args:
        chemistry (Molecule or Structure): The chemical to be optimized
        prev_calc_data (dict): Start an optimization from a previous state,
            output from berny
        logfile (str): File path specifying where Berny logging information
            should be sent. By default, this is "berny.log".
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

    def __init__(self, chemistry, prev_calc_data=None, logfile="berny.log",
                 verbosity=None, transition_state=False, max_steps=250,
                 max_gradient=4.5e-4, rms_gradient=3.0e-4, max_step_size=1.8e-3,
                 rms_step_size=1.2e-3, trust=0.3, min_trust=1e-6, dihedral=True,
                 weak_dihedral=False):

        self.initial_chemistry = chemistry
        self.chemistry = self.initial_chemistry

        if isinstance(self.chemistry, Structure):
            self.lattice = self.chemistry.lattice.matrix

        elif isinstance(self.chemistry, Molecule):
            self.lattice = None
        else:
            raise ValueError("chemistry must be either a Structure or Molecule "
                             "object!")

        geom = geomlib.Geometry([str(e.specie) for e in self.chemistry.sites],
                                [e.coords for e in self.chemistry],
                                lattice=self.lattice)

        self.logger = BernyLogger(verbosity=verbosity, logfile=logfile)
        self.transition_state = transition_state

        self.params = {"gradientmax": max_gradient,
                       "gradientrms": rms_gradient,
                       "stepmax": max_step_size,
                       "steprms": rms_step_size,
                       "trust": trust,
                       "min_trust": min_trust,
                       "dihedral": dihedral,
                       "superweakdih": weak_dihedral}

        self.berny = Berny(geom, self.logger, debug=True,
                           restart=prev_calc_data, maxsteps=max_steps,
                           transition_state=self.transition_state,
                           params=self.params)

        self.state = self.berny.state
        self.max_steps = max_steps

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

        state_dict = self.berny.send((energy, gradients))
        self.state = State(geom=state_dict["geom"],
                           coords=state_dict["coords"],
                           trust=state_dict["trust"],
                           hessian=state_dict["H"],
                           weights=state_dict["weights"],
                           future=state_dict["future"],
                           params=state_dict["params"],
                           first=state_dict["first"])

        new_geom = self.state.geom
        if isinstance(self.initial_chemistry, Structure):
            self.chemistry = Structure(lattice=self.lattice,
                                       species=new_geom.species,
                                       coords=new_geom.coords,
                                       charge=self.initial_chemistry.charge)
        elif isinstance(self.initial_chemistry, Molecule):
            self.chemistry = Molecule(species=new_geom.species,
                                      coords=new_geom.coords,
                                      charge=self.initial_chemistry.charge,
                                      spin_multiplicity=self.initial_chemistry.spin_multiplicity)

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

    def set_hessian_exact(self, gradients, hessian):
        """
        Update the state of the optimizer with an exact Hessian.

        Args:
            gradients (np.ndarray): Gradient vector in Cartesian coordinates.
            hessian (np.ndarray): Hessian matrix in Cartesian coordinates.

        Returns:
            None
        """

        self.berny.update_hessian_exact(gradients, hessian)
        self.state = self.berny.state
