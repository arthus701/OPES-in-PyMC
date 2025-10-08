from pymc.sampling import jax as pmj

from inference_loop import inference_loop

from model import myModel

# import arviz as az

# Monkey-patch the original inference loop with our own
pmj._blackjax_inference_loop = inference_loop

n_chains = 4

with myModel:
    idata = pmj.sample_blackjax_nuts(
        100_000,
        # 1_000_000,      # Hanna's value
        tune=0,
        chains=n_chains,
        progressbar=True,
        target_accept=0.95,
        random_seed=1301512,
    )

idata.to_netcdf('./samples.nc')
# print(az.summary(idata))
