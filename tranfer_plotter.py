# General imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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
                              c=np.linalg.norm(state_history[:,3:] / 1000,
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

transfer = pd.read_csv("data_constant_dt_solar_2/trajectory_0.csv", index_col=0)
transfer = transfer.to_dict(orient="list")
transfer = np.array(list(transfer.values()))
plot_transfer(transfer, celestial_bodies=True)
