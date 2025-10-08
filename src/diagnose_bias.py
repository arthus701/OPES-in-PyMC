import numpy as np

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

import arviz as az

from model import energy_function

idata = az.from_netcdf('./samples.nc')
delta_F = idata.sample_stats['delta_F'].values
x_samps = idata.posterior['x'].values
NUM_LAMBDA = delta_F.shape[2]
SIGMA = 0.5

chain = 0

fig, ax = plt.subplots(
    1, 1,
    figsize=(13, 6),
)

bnds = ax.get_position().bounds
colax_width = bnds[2]
sliderloc = ax.get_position().bounds
sax = fig.add_axes([sliderloc[0], sliderloc[1]-0.1, colax_width, 0.02])

iteration_sel = Slider(
    sax,
    'Iteration',
    # max(knots.min(), knots_old.min()),
    0,
    # delta_F.shape[1]-1,
    999,
    valinit=0,
    valstep=1,
    valfmt='%i',
)

# ax.set_ylim(-12, 3)

x_array = np.linspace(-3, 3, 1001)

gaussian_centers = -3 + 6 * np.arange(NUM_LAMBDA) / (NUM_LAMBDA - 1)


def get_bias_function(it):
    delta_F_value = delta_F[chain, it, :]

    V = np.exp(
        -(x_array[:, None] - gaussian_centers[None, :]) ** 2 / (2 * SIGMA ** 2)
        + delta_F_value[None, :]
    )

    return np.log(np.mean(V, axis=-1))


line, = ax.plot(
    x_array,
    get_bias_function(0)
    # + energy_function(x_array, x_samps[chain, 0, 1]),
)
line_2, = ax.plot(
    x_array,
    get_bias_function(0) - energy_function((x_array, x_samps[chain, 0, 1])),
)
x_at = ax.axvline(
    x_samps[chain, 0, 0],
    color='grey',
)


def update(val):
    line.set_ydata(
        get_bias_function(int(iteration_sel.val))
        # + energy_function(x_array, x_samps[chain, iteration_sel.val, 1])
    )
    line_2.set_ydata(
        get_bias_function(int(iteration_sel.val))
        - energy_function((x_array, x_samps[chain, iteration_sel.val, 1]))

    )
    x_at.set_xdata([x_samps[chain, iteration_sel.val, 0]])
    fig.canvas.draw_idle()


iteration_sel.on_changed(update)


def on_key(event):
    if event.key == 'right':
        new_val = min(
            iteration_sel.val + iteration_sel.valstep,
            iteration_sel.valmax,
        )
        iteration_sel.set_val(new_val)
    elif event.key == 'left':
        new_val = max(
            iteration_sel.val - iteration_sel.valstep,
            iteration_sel.valmin,
        )
        iteration_sel.set_val(new_val)


# Connect the event
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
