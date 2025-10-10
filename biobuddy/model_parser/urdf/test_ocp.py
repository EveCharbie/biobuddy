"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""

from bioptim import (
    OptimalControlProgram,
    DynamicsOptions,
    BoundsList,
    InitialGuessList,
    ObjectiveList,
    ObjectiveFcn,
    OdeSolver,
    OdeSolverBase,
    Solver,
    TorqueBiorbdModel,
    ControlType,
    PhaseDynamics,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = False,
    control_type: ControlType = ControlType.CONSTANT,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    control_type: ControlType
        The type of the controls

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = TorqueBiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q")

    # DynamicsOptions
    dynamics = DynamicsOptions(
        ode_solver=ode_solver,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

    # Path bounds
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0  # Start at 0...
    x_bounds["q"][:, -1] = 3.14 / 2  # ...but end with all joints 90 degrees rotated
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0  # Start and end without any velocity

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot

    # Define control path bounds
    n_tau = bio_model.nb_tau
    u_bounds = BoundsList()
    # u_bounds["tau"] = [-100] * n_tau, [100] * n_tau  # Limit the strength of the pendulum to (-100 to 100)...
    # u_bounds["tau"][1, :] = 0  # ...but remove the capability to actively rotate

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    u_init = InitialGuessList()
    u_init["tau"] = [0] * n_tau

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        control_type=control_type,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="kuka_lwr.bioMod", final_time=1.0, n_shooting=200, n_threads=28)

    # --- Solve the ocp --- #
    # Default is OnlineOptim.MULTIPROCESS on Linux, OnlineOptim.MULTIPROCESS_SERVER on Windows and None on MacOS
    # To see the graphs on MacOS, one must run the server manually (see resources/plotting_server.py)
    sol = ocp.solve(Solver.IPOPT())

    # --- Animate the solution --- #
    # viewer = "bioviz"
    viewer = "pyorerun"
    sol.animate(viewer=viewer)


if __name__ == "__main__":
    main()
