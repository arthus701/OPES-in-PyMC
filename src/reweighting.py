
import numpy as np
import arviz as az

from inference_loop import LAMBDAS, ZERO_TEMP

rng = np.random.default_rng(130118)

iData = az.from_netcdf('./samples.nc')

x_samps = iData.posterior['x'].values
warmup = int(round(x_samps.shape[1] * 0.1))

v_k = iData.sample_stats['bias_value'].values
logp = iData.sample_stats['lp'].values
lambda_idx = 0
delta_u = -LAMBDAS[None, None, :] / ZERO_TEMP * logp[:, :, None]

weights = np.exp(-delta_u[:, warmup:, ] + v_k[:, warmup:, None])
weights = weights / np.sum(weights, axis=1)[:, None]

weights_at = weights[:, :, lambda_idx]

x_resampled = np.zeros(
    (weights.shape[0], weights.shape[1], x_samps.shape[-1])
)

# XXX Reweighting is turned off, as during experimentation the weights
# sometimes contained nans, breaking the script
for it, weights_i in enumerate(weights_at):
    indices = rng.choice(
        np.arange(warmup, weights.shape[1] + warmup),
        size=weights.shape[1],
        replace=True,
        p=weights_i,
    ).astype(int)

    x_resampled[it] = np.copy(x_samps[it, indices, :])
