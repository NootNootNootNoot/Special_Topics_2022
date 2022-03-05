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

transfer = pd.read_csv("data_10000/trajectory_0.csv", index_col=0)
planets = transfer[["Mars", "Earth"]]
transfer = transfer.drop(columns = ["Mars", "Earth"])
initial_time = float(transfer.columns[0])
final_time = float(transfer.columns[-1])
print(np.size(transfer.columns))
transfer = transfer.to_dict(orient="list")
Util.plot_transfer(transfer, initial_time, final_time, "Mars")
