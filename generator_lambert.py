
# General imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Tudatpy imports
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup

# Problem-specific imports
import Utilities as Util

# Vehicle settings
vehicle_mass = 4.0E3
# Fixed parameters
minimum_mars_distance = 5.0E7
# Time since 'departure from Earth CoM' at which propagation starts (and similar
# for arrival time)
time_buffer = 30.0*constants.JULIAN_DAY

for i in tqdm(range(100000)):

    # Time at which to start propagation
    initial_time = 303.3622685185*constants.JULIAN_DAY + np.random.uniform(-303.3622685185*constants.JULIAN_DAY,
                                                                           22*365*constants.JULIAN_DAY, 1)
    transfer_time = 385.03125*constants.JULIAN_DAY + np.random.uniform(-385.03125*constants.JULIAN_DAY/2,
                                                                       385.03125*constants.JULIAN_DAY, 1)
    final_time = initial_time + transfer_time
    target_planet = "Mars"
    initial_propagation_time = initial_time + time_buffer
    final_propagation_time = final_time - time_buffer

    # Define settings for celestial bodies
    bodies_to_create = ['Earth',
                        target_planet,
                        'Sun']
    # Define coordinate system
    global_frame_origin = 'SSB'
    global_frame_orientation = 'ECLIPJ2000'
    # Create body settings
    body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                                global_frame_origin,
                                                                global_frame_orientation)
    # Create bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create vehicle object and add it to the existing system of bodies
    bodies.create_empty_body('Vehicle')
    bodies.get_body('Vehicle').mass = vehicle_mass

    initial_state, final_state, keplerian_elements = Util.get_lambert_problem_result_fast(bodies, target_planet,
                                                   initial_time,
                                                   final_time)

    states_E, state_T = Util.get_environment_states(initial_time,
                                                    final_time,
                                                    target_planet)


    state_lambert = {
        "initial_state": initial_state,
        "final_state": final_state,
        "keplerian_elements": keplerian_elements,
        "Earth": states_E,
        target_planet: state_T,
        "predictor_settings": np.array([initial_time[0],
                                         final_time[0],
                                         time_buffer,
                                         vehicle_mass,
                                         minimum_mars_distance, 0])
    }
    df = pd.DataFrame(state_lambert)
    input = pd.DataFrame()
    df.to_csv(f"data_lambert_cut_2/trajectory_{i}.csv")
