
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

store_initial_final_states_only = False

# Vehicle settings
vehicle_mass = 4.0E3

dt = 14400

# Define settings for celestial bodies
bodies_to_create = ["Sun", "Saturn", "Mercury", "Uranus", "Venus", "Neptune", "Pluto", 'Earth','Mars','Jupiter',"Moon"]
# Define coordinate system
global_frame_origin = 'SSB'
global_frame_orientation = 'ECLIPJ2000'
# Create body settings
body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                            global_frame_origin,
                                                            global_frame_orientation)
# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)
initial_propagation_time = 0
final_propagation_time = 100*dt
au = 149597870700

for i in tqdm(range(1)):

    initial_state = np.array([np.random.uniform(0.5*au, 5*au),
                              np.random.uniform(0.5*au, 5*au),
                              np.random.uniform(0.5*au, 5*au),
                              np.random.uniform(0, 40E3),
                              np.random.uniform(0, 40E3),
                              np.random.uniform(0, 40E3)])
    initial_state = initial_state * np.random.choice([-1,1], 6)

    # initial_state = np.array([1,1,1,
    #                           0,0,27E3])
    # initial_state[:3] /= np.linalg.norm(initial_state[:3])/au

    # Create vehicle object and add it to the existing system of bodies
    bodies.create_empty_body('Vehicle')
    bodies.get_body('Vehicle').mass = vehicle_mass

    # Retrieve termination settings
    termination_settings = propagation_setup.propagator.time_termination( final_propagation_time )

    # Create integrator settings
    integrator = propagation_setup.integrator
    integrator_settings = lambda x: integrator.adams_bashforth_moulton(x, dt, dt, dt, np.inf, np.inf)
    integrator_settings = integrator_settings(initial_propagation_time)

    state_lambert = Util.propagate_trajectory(initial_propagation_time,
                                                        termination_settings,
                                                        bodies, initial_state,
                                                        integrator_settings,
                                                        propagation_setup.propagator.cowell).state_history
    # Util.plot_transfer(state_lambert, initial_time,
    #                    final_time, target_planet)

    df = pd.DataFrame(state_lambert)
    input = pd.DataFrame()
    df.to_csv(f"data_test/trajectory_{i}.csv")
