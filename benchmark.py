import numpy as np
import tensorflow as tf
import joblib
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

sns.set_theme()
sns.set_palette("rocket")

def plot_transfer(state_history,
                  figname = f"Transfer trajectory",
                  celestial_bodies = True,
                  cbar_plot = True,
                  dark_theme = True,
                  save = False,
                  line = False,
                  ax = None,
                  fig = None):

    state_history = state_history.reshape(-1, 6)

    if ax == None or fig == None:
        fig = plt.figure()
        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, projection="3d")
        plt.axis('off')

    if dark_theme:
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

    else:
        ax.set_facecolor("white")

    if cbar_plot:
        transfer = ax.scatter(state_history[:, 0],
                              state_history[:, 1],
                              state_history[:, 2],
                              c=np.round(np.linalg.norm(state_history[:,3:] / 1000,
                                           axis=1),2),
                              cmap="coolwarm",
                              zorder=12)
        cbar = plt.colorbar(transfer, ax=ax)
        cbar.set_label('Velocity Magnitude [km/s]')

        if dark_theme:
            cbar.ax.tick_params(color="white")
            cbar.ax.set_yticklabels(labels=cbar.ax.get_yticks(), color="white",
                                    fontsize=22)
            cbar.set_label('Velocity Magnitude [km/s]', color="white", fontsize=26)


    else:
        ax.scatter(state_history[:, 0],
                   state_history[:, 1],
                   state_history[:, 2],
                   c=np.linalg.norm(state_history[:, 3:] / 1000,
                                    axis=1),
                   cmap="coolwarm",
                   zorder=12)
    if line:
        ax.plot(state_history[:, 0],
                state_history[:, 1],
                state_history[:, 2],
                c="r",zorder = 4, ls = "--")

    if celestial_bodies:

        # Scaled sun model
        R_S = 696340e3 * 50
        u, v = np.mgrid[-np.pi:np.pi:20j, 0:np.pi:20j]
        xs = R_S * np.cos(u) * np.sin(v)
        ys = R_S * np.sin(u) * np.sin(v)
        zs = R_S * np.cos(v)
        ax.plot_wireframe(xs, ys, zs, color="yellow", linewidth=0.5, zorder=-1)


        ax.set_box_aspect((np.ptp(state_history[:, 0]),
                           np.ptp(state_history[:, 1]),
                           np.ptp(zs)))
        fig.canvas.manager.set_window_title(figname)

        if save:
            fig.savefig(f"{figname}.png", facecolor=fig.get_facecolor())

    return

np.set_printoptions(precision=3)
NN_pos = tf.keras.models.load_model("model_3/model_pos")
NN_vel = tf.keras.models.load_model("model_3/model_vel")
au = 149597870700
x_size = 6*6
scaler_x = joblib.load("model_3/scaler_x")

# direct to folder
folder_path = r"data_test/"

trajectories = np.zeros(2, dtype=object)
integrator = np.copy(trajectories)

counter = 0
for file in tqdm(os.listdir(folder_path)[:2]):
    print(file)
    contents = np.genfromtxt(folder_path + file, delimiter=",")
    contents[1:4, 1:] /= au
    contents[4:, 1:] /= 10 ** 3
    trajectories[counter] = contents[1:,1:7].reshape(1, -1, order = 'F')[0]
    integrator[counter] = contents[1:,1:].reshape(1, -1, order = 'F')[0]
    counter += 1

integrator = np.concatenate(integrator).reshape(2, -1 ,6)
x = np.concatenate(trajectories).reshape(-1,x_size)
x_scaled = scaler_x.transform(x)

y_size = 100
y = np.zeros((2, y_size, 6))
for i in tqdm(range(y_size)):
    y[:,i,:3] = NN_pos.predict(x_scaled)
    y[:,i,3:] = NN_vel.predict(x_scaled)
    x = np.array([np.roll(x[i], -6) for i in range(np.shape(x)[0])])
    x[:,-6:] = y[:,i,:]
    x_scaled = scaler_x.transform(x)

y[:,:,:3] *= au
y[:,:,3:] *= 10**3
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, projection="3d")
plt.axis('off')
for i,trajectory in enumerate(y):
    if i ==0:
        plot_transfer(trajectory, celestial_bodies=False, ax=ax, fig=fig)