import numpy as np

from matplotlib import pyplot as plt

from reweighting import weights
from inference_loop import NUM_TEMPS, TEMPS

n_eff_lambda = np.sum(weights, axis=1)**2 / np.sum(weights**2, axis=1)
n_eff_lambda /= weights.shape[1]
target = 1 / NUM_TEMPS

fig, ax = plt.subplots(
    1, 1,
    figsize=(13, 6),
)
for it in range(4):
    ax.plot(
        TEMPS,
        n_eff_lambda[it],
        label=f'Chain #{it+1}',
    )

ax.axhline(target, color='grey', label='Target', ls='--')


ax.set_xscale('log')
ax.set_xlabel('Temperature')
ax.set_yscale('log')
ax.set_ylabel(r'$n_\text{eff} / n$')

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    '../diagnosis_neff.png',
    bbox_inches='tight',
    pad_inches=0.05,
    dpi=300,
)

plt.show()
