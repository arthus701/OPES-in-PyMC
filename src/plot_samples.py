from matplotlib import pyplot as plt
from matplotlib import colors

import numpy as np
import arviz as az

from model import energy_function

rng = np.random.default_rng(130118)

x, dx = np.linspace(-3, 3, 1000, retstep=True)
y, dy = np.linspace(-3, 3, 1000, retstep=True)

X, Y = np.meshgrid(x, y, indexing="xy")

U = -energy_function((X, Y))
P = np.exp(U)
P /= np.sum(P) * dx * dy

norm = colors.Normalize(vmin=P.min(), vmax=P.max())

idata = az.from_netcdf('./samples.nc')

cutout = 100
bias_values = idata.sample_stats['bias_value'].values
delta_F = idata.sample_stats['delta_F'].values
x_samps = idata.posterior['x'].values

n_chains = x_samps.shape[0]

weights = np.exp(bias_values[:, cutout:])
weights = weights / np.sum(weights, axis=-1)[:, None]

x_resampled = np.zeros((*weights.shape, x_samps.shape[-1]))

# XXX Reweighting is turned off, as during experimentation the weights
# sometimes contained nans, breaking the script
for it, weights_i in enumerate(weights):
    indices = rng.choice(
        np.arange(cutout, weights.shape[1] + cutout),
        size=weights.shape[1],
        replace=True,
        p=weights_i,
    ).astype(int)
    x_resampled[it] = np.copy(x_samps[it, indices, :])

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
        bins=20,
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
for it, _x in enumerate(x_resampled):
    axs[it+1, 1].scatter(
        _x[:, 0],
        _x[:, 1],
        alpha=0.5,
    )

fig.tight_layout()

plt.show()

fig_2, ax_2 = plt.subplots(1, 1, figsize=(10, 5))
ax_2.set_title("Bias over iterations")
ax_2.plot(
    bias_values.T,
)
# ax_2.scatter(
#     x_samps[:, :, 0].flatten(),
#     idata.sample_stats['lp'].values.flatten(),
# )
# ax_2.set_yscale('log')
ax_2.set_xlabel('Iteration number')
ax_2.set_ylabel('Bias value')

plt.show()
