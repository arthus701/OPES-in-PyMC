from matplotlib import pyplot as plt
from matplotlib import colors

import numpy as np

from model import energy_function
from reweighting import x_samps, x_resampled

x, dx = np.linspace(-3, 3, 1000, retstep=True)
y, dy = np.linspace(-3, 3, 1000, retstep=True)

X, Y = np.meshgrid(x, y, indexing="xy")

U = -energy_function((X, Y))
P = np.exp(U)
P /= np.sum(P) * dx * dy

norm = colors.Normalize(vmin=P.min(), vmax=P.max())

n_chains = x_samps.shape[0]

fig, axs = plt.subplots(
    1 + n_chains, 2,
    sharex=True,
    sharey=True,
    figsize=(10, 15),
)

for ax in axs[0]:
    ax.pcolormesh(
        X,
        Y,
        -np.log(P),
        norm=norm,
        cmap='viridis_r',
        # levels=21,
    )

axs[0, 0].set_title("Sampled")
# for it, _x in enumerate(x_samps):
#     axs[it+1, 0].scatter(
#         _x[:, 0],
#         _x[:, 1],
#         alpha=0.05,
#     )
for it, _x in enumerate(x_samps):
    hist, xedges, yedges = np.histogram2d(
        *_x.T,
        bins=100,
        range=[[-3, 3], [-3, 3]],
        density=True,
    )

    xpos = (xedges[:-1] + xedges[1:]) / 2
    ypos = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xpos, ypos, indexing='ij')

    # norm = colors.Normalize(vmin=hist.min(), vmax=hist.max())
    pcm = axs[it+1, 0].pcolormesh(
        X,
        Y,
        -np.log(hist + 1e-12),
        cmap='viridis_r',
        norm=norm,
    )

axs[0, 1].set_title("Re-Sampled")
# for it, _x in enumerate(x_resampled):
#     axs[it+1, 1].scatter(
#         _x[:, 0],
#         _x[:, 1],
#         alpha=0.5,
#     )
for it, _x in enumerate(x_resampled):
    hist, xedges, yedges = np.histogram2d(
        *_x.T,
        bins=100,
        range=[[-3, 3], [-3, 3]],
        density=True,
    )

    xpos = (xedges[:-1] + xedges[1:]) / 2
    ypos = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xpos, ypos, indexing='ij')

    # norm = colors.Normalize(vmin=hist.min(), vmax=hist.max())
    pcm = axs[it+1, 1].pcolormesh(
        X,
        Y,
        -np.log(hist + 1e-12),
        cmap='viridis_r',
        norm=norm,
    )

fig.tight_layout()

plt.show()
