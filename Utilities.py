# General imports
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter as sf

# plot visuals
sns.set_theme()
sns.color_palette("rocket")

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import environment

# Load spice kernels
spice_interface.load_standard_kernels()

def get_environment_states(initial_time, final_time, target_body):

    state_E = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name="Earth",
        observer_body_name="Sun",
        reference_frame_name='ECLIPJ2000',
        aberration_corrections="NONE",
        ephemeris_time=initial_time)

    state_T = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name=target_body,
        observer_body_name="Sun",
        reference_frame_name='ECLIPJ2000',
        aberration_corrections="NONE",
        ephemeris_time=final_time)

    return state_E, state_T





def get_termination_settings(termination_time,
                             target_planet:str,
                             minimum_mars_distance: float,
                             time_buffer: float) \
        -> tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings:
    """
    Get the termination settings for the simulation.

    Termination settings currently include:
    - simulation time (propagation stops if it is greater than the one provided by the hodographic trajectory)
    - distance to Mars (propagation stops if the relative distance is lower than the target distance)

    Parameters
    ----------
    trajectory_parameters : list[floats]
        List of trajectory parameters.
    minimum_mars_distance : float
        Minimum distance from Mars at which the propagation stops.
    time_buffer : float
        Time interval between the simulation start epoch and the beginning of the hodographic trajectory.

    Returns
    -------
    hybrid_termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object.
    """
    # Create single PropagationTerminationSettings objects
    # Time
    final_time = termination_time
    time_termination_settings = propagation_setup.propagator.time_termination(
        final_time,
        terminate_exactly_on_final_condition=False)
    # Altitude
    relative_distance_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.relative_distance('Vehicle', target_planet),
        limit_value=minimum_mars_distance,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False)
    # Define list of termination settings
    termination_settings_list = [time_termination_settings,
                                 relative_distance_termination_settings]
    # Create termination settings object
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                  fulfill_single_condition=True)
    return hybrid_termination_settings

def plot_transfer(state_history,
                  initial_time,
                  final_time,
                  target_planet,
                  figname = f"Transfer trajectory",
                  celestial_bodies = True,
                  global_frame_orientation = 'ECLIPJ2000',
                  cbar_plot = True,
                  dark_theme = True,
                  save = False,
                  line = False):

    fig = plt.figure()
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, projection="3d")
    plt.axis('off')

    if dark_theme:
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

    else:
        ax.set_facecolor("white")

    if cbar_plot:
        transfer = ax.scatter(np.array(list(state_history.values()))[:, 0],
                              np.array(list(state_history.values()))[:, 1],
                              np.array(list(state_history.values()))[:, 2],
                              c=np.linalg.norm(np.array(list(state_history.values()))[:, 3:] / 1000,
                                           axis=1),
                              cmap="coolwarm",
                              zorder=12)
        cbar = plt.colorbar(transfer, ax=ax)
        cbar.set_label('Velocity Magnitude [km/s]')

        if dark_theme:
            cbar.ax.tick_params(color="white")
            cbar.ax.set_yticklabels(labels=cbar.ax.get_yticks(), color="white")
            cbar.set_label('Velocity Magnitude [km/s]', color="white")


    else:
        ax.scatter(np.array(list(state_history.values()))[:, 0],
                   np.array(list(state_history.values()))[:, 1],
                   np.array(list(state_history.values()))[:, 2],
                   c="r",
                   zorder=12)
    if line:
        ax.plot(np.array(list(state_history.values()))[:, 0],
                np.array(list(state_history.values()))[:, 1],
                np.array(list(state_history.values()))[:, 2],
                c="r",zorder = 4, ls = "--")

    if celestial_bodies:

        # Scaled sun model
        R_S = 696340e3 * 50
        u, v = np.mgrid[-np.pi:np.pi:20j, 0:np.pi:20j]
        xs = R_S * np.cos(u) * np.sin(v)
        ys = R_S * np.sin(u) * np.sin(v)
        zs = R_S * np.cos(v)
        ax.plot_wireframe(xs, ys, zs, color="yellow", linewidth=0.5, zorder=-1)

        # Scaled Earth model
        initial_state_E = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name="Earth",
            observer_body_name="Sun",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=initial_time)
        R_E = R_S / 3
        xe = R_E * np.cos(u) * np.sin(v) + initial_state_E[0]
        ye = R_E * np.sin(u) * np.sin(v) + initial_state_E[1]
        ze = R_E * np.cos(v) + initial_state_E[2]
        ax.plot_wireframe(xe, ye, ze, color="royalblue", linewidth=0.5, zorder=-1)

        # Scaled Mars model
        initial_state_M = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name=target_planet,
            observer_body_name="Sun",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=final_time)
        R_M = R_S / 3.5
        xm = R_M * np.cos(u) * np.sin(v) + initial_state_M[0]
        ym = R_M * np.sin(u) * np.sin(v) + initial_state_M[1]
        zm = R_M * np.cos(v) + initial_state_M[2]
        ax.plot_wireframe(xm, ym, zm, color="orange", linewidth=0.5, zorder=-1)

        ax.set_box_aspect((
            np.ptp(np.array(list(state_history.values()))[:, 0]),
            np.ptp(np.array(list(state_history.values()))[:, 1]),
            np.ptp(zs)
        ))
        fig.canvas.manager.set_window_title(figname)

        if save:
            fig.savefig(f"{figname}.png", facecolor=fig.get_facecolor())

    return

def plot_pos_error(state_difference,
                   figure = True,
                   axis = True,
                   smooth = False,
                   cut = False,
                   buffer = 0,
                   smoother = 5,
                   figname = f"position_difference"):

    if figure or axis:
        fig, ax = plt.subplots(1, 1, figsize=(20, 17))

    else:
        fig = figure
        ax = axis

    errors = np.linalg.norm(np.array(list(state_difference.values()))[:,:3], axis = 1)
    t = (np.array(list(state_difference.keys())) - np.array(list(state_difference.keys()))[0]) / \
        constants.JULIAN_DAY

    if smooth:
        for j in range(smoother):
            z = np.abs(stats.zscore(errors))
            errors = errors[np.where(z<3)]
            t = t[np.where(z<3)]

    if cut:
        errors = errors[np.max(np.where(t <= buffer)):
                        np.max(np.where(t[-1] - t >= buffer))]
        t = t[np.max(np.where(t <= buffer)):
                     np.max(np.where(t[-1] - t >= buffer))]


    ax.scatter(t, errors)
    ax.plot(t, errors, ls="--")

    ax.grid(zorder=-2)
    ax.set_yscale("log")

    fig.canvas.manager.set_window_title(figname)
    return t, errors

def get_noise(y, poly_order = 3):

    fit = sf(y, int(np.ceil(np.min((np.size(y), poly_order*5)))), poly_order)
    deviations = y - fit

    z = np.abs(stats.zscore(deviations))
    noise = np.mean(z)

    return noise, fit

# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_problem_result_fast(
        bodies: environment.SystemOfBodies,
        target_body: str,
        departure_epoch: float,
        arrival_epoch: float) -> environment.Ephemeris:

    # Gravitational parameter of the Sun
    central_body_gravitational_parameter = bodies.get_body("Sun").gravitational_parameter

    # Set initial and final positions for Lambert targeter
    initial_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name="Earth",
        observer_body_name="Sun",
        reference_frame_name='ECLIPJ2000',
        aberration_corrections="NONE",
        ephemeris_time=departure_epoch)

    final_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name=target_body,
        observer_body_name="Sun",
        reference_frame_name='ECLIPJ2000',
        aberration_corrections="NONE",
        ephemeris_time=arrival_epoch)

    # Create Lambert targeter
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3], final_state[:3], arrival_epoch - departure_epoch, central_body_gravitational_parameter);

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = initial_state
    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()
    lambert_arc_final_state = final_state
    lambert_arc_final_state[3:] = lambertTargeter.get_arrival_velocity()

    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(lambert_arc_initial_state,
                                                                       central_body_gravitational_parameter)

    return lambert_arc_initial_state, lambert_arc_final_state , lambert_arc_keplerian_elements


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_problem_result(
        bodies: environment.SystemOfBodies,
        target_body: str,
        departure_epoch: float,
        arrival_epoch: float,
        return_more = False) -> environment.Ephemeris:

    """"
    This function solved Lambert's problem for a transfer from Earth (at departure epoch) to
    a target body (at arrival epoch), with the states of Earth and the target body defined
    by ephemerides stored inside the SystemOfBodies object (bodies). Note that this solver
    assumes that the transfer departs/arrives to/from the center of mass of Earth and the target body

    Parameters
    ----------
    bodies : Body objects defining the physical simulation environment

    target_body : The name (string) of the body to which the Lambert arc is to be computed

    departure_epoch : Epoch at which the departure from Earth's center of mass is to take place

    arrival_epoch : Epoch at which the arrival at he target body's center of mass is to take place

    Return
    ------
    Ephemeris object defining a purely Keplerian trajectory. This Keplerian trajectory defines the transfer
    from Earth to the target body according to the inputs to this function. Note that this Ephemeris object
    is valid before the departure epoch, and after the arrival epoch, and simply continues (forwards and backwards)
    the unperturbed Sun-centered orbit, as fully defined by the unperturbed transfer arc
    """

    # Gravitational parameter of the Sun
    central_body_gravitational_parameter = bodies.get_body("Sun").gravitational_parameter

    # Set initial and final positions for Lambert targeter
    initial_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name="Earth",
        observer_body_name="Sun",
        reference_frame_name='ECLIPJ2000',
        aberration_corrections="NONE",
        ephemeris_time=departure_epoch)

    final_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name=target_body,
        observer_body_name="Sun",
        reference_frame_name='ECLIPJ2000',
        aberration_corrections="NONE",
        ephemeris_time=arrival_epoch)

    # Create Lambert targeter
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3], final_state[:3], arrival_epoch - departure_epoch, central_body_gravitational_parameter);

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = initial_state
    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()
    lambert_arc_final_state = final_state
    lambert_arc_final_state[:3] = lambertTargeter.get_arrival_velocity()

    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(lambert_arc_initial_state,
                                                                       central_body_gravitational_parameter)

    # Setup Keplerian ephemeris model that describes the Lambert arc
    kepler_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(lambert_arc_keplerian_elements, departure_epoch,
                                              central_body_gravitational_parameter), "")

    if return_more:
        return kepler_ephemeris, lambert_arc_initial_state, lambert_arc_final_state , lambert_arc_keplerian_elements
    else:
        return kepler_ephemeris

# STUDENT CODE TASK REMOVE - full function (except signature and return)
def get_perturbed_propagator_settings(
        bodies: environment.SystemOfBodies,
        initial_state: np.array,
        termination_settings: float,
        propagator: propagation_setup.propagator):

    # Create Radiation coefficients interface
    rad_coefficient = 1.2
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", 20, rad_coefficient)
    environment_setup.add_radiation_pressure_interface(
        bodies, "Vehicle", radiation_pressure_settings)

    # Define accelerations acting on vehicle.
    acceleration_settings_on_Vehicle = dict(
        Sun=
        [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.cannonball_radiation_pressure()
        ],
        # Venus=
        # [
        #     propagation_setup.acceleration.point_mass_gravity()
        # ],
        # Earth=
        # [
        #     propagation_setup.acceleration.point_mass_gravity()
        # ],
        # Moon=
        # [
        #     propagation_setup.acceleration.point_mass_gravity()
        # ],
        # Mars=
        # [
        #     propagation_setup.acceleration.point_mass_gravity()
        # ],
        # Jupiter=
        # [
        #     propagation_setup.acceleration.point_mass_gravity()
        # ],
        # Saturn=
        # [
        #     propagation_setup.acceleration.point_mass_gravity()
        # ]
    )

    # Define bodies that are propagated, and their central bodies of propagation.
    bodies_to_propagate = ['Vehicle']
    central_bodies = ['Sun']

    # Create global accelerations dictionary.
    acceleration_settings = {'Vehicle': acceleration_settings_on_Vehicle}

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    # Define required outputs
    dependent_variables_to_save = [propagation_setup.dependent_variable.total_acceleration("Vehicle")]

    # Create propagation settings.
    propagator_settings = propagation_setup.propagator.translational(
                                central_bodies,
                                acceleration_models,
                                bodies_to_propagate,
                                initial_state,
                                termination_settings,
                                propagator,
                                output_variables = dependent_variables_to_save
                            )

    return propagator_settings

# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def propagate_trajectory(
        initial_time: float,
        termination_settings: float,
        bodies: environment.SystemOfBodies,
        initial_state: np.array,
        integrator_settings: propagation_setup.integrator,
        propagator: propagation_setup.propagator,
        initial_state_correction=np.array([0, 0, 0, 0, 0, 0])) -> numerical_simulation.SingleArcSimulator:


    # Compute initial state along Lambert arc (and apply correction if needed)
    lambert_arc_initial_state = initial_state

    propagator_settings = get_perturbed_propagator_settings(
        bodies, lambert_arc_initial_state, termination_settings, propagator)


    # Propagate forward/backward perturbed/unperturbed arc and save results to files
    dynamics_simulator = numerical_simulation.SingleArcSimulator(bodies, integrator_settings, propagator_settings,
                                                                 print_dependent_variable_data = False)

    return dynamics_simulator