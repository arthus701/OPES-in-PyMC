from functools import partial

import jax
import jax.numpy as jnp
import blackjax

from typing import NamedTuple
from blackjax.mcmc.hmc import HMCState

# Use this for toying with other integrators
# from blackjax.mcmc.integrators import yoshida

NUM_LAMBDA = 201
MAX_TEMP = 1000
ZERO_TEMP = 1
# TEMPS = jnp.linspace(1, MAX_TEMP, NUM_LAMBDA)
# LAMBDAS = 1 / TEMPS - 1
LAMBDAS = jnp.linspace(1 / MAX_TEMP, 1 / ZERO_TEMP, NUM_LAMBDA)[::-1] \
    - 1 / ZERO_TEMP
TEMPS = 1 / (LAMBDAS + 1 / ZERO_TEMP)
INTEGRATION_STEPS = 10
STEPSIZE = 0.01
HMC_STEPS = 10
BIAS_CAP = None


class BiasState(NamedTuple):
    state: HMCState
    delta_F_nominator_sum: jax.Array
    delta_F_denominator_sum: jax.Array
    delta_F: jax.Array
    bias_value: float
    count: float


def bias_potential(u, delta_F):
    # NOTE: u = -logp
    delta_u = LAMBDAS / ZERO_TEMP * u
    V = jnp.exp(
        -delta_u + delta_F
    )
    return -jnp.log(jnp.sum(V) / NUM_LAMBDA)


def update_delta_F(
    u,
    delta_F_nominator_sum,
    delta_F_denominator_sum,
    delta_F,
    potential,
):
    delta_F_nominator_sum += jnp.exp(
        -LAMBDAS / ZERO_TEMP * u + potential * jnp.ones(NUM_LAMBDA)
    )

    delta_F_denominator_sum += jnp.exp(potential)

    delta_F = -jnp.log(
        delta_F_nominator_sum / delta_F_denominator_sum
    )

    delta_F = jnp.clip(
        delta_F,
        min=None,
        max=BIAS_CAP,
    )

    return delta_F_nominator_sum, delta_F_denominator_sum, delta_F


# This is the replacement inference_loop
# A reference implementation can be found here:
# https://github.com/pymc-devs/pymc/blob/
# 340e403b8813ab5f3699a476cc828cc92c4f9d50/
# pymc/sampling/jax.py#L250
def inference_loop(
    seed, init_position, logp_fn, draws, tune, target_accept,
    **adaptation_kwargs
):
    # Ignore passed algorithm kwarg and always use hmc (for now)
    adaptation_kwargs.pop("algorithm", "nuts")
    algorithm = blackjax.hmc

    # Set up initial state, init_position is passed from outside
    # Default should be uniform
    grad_fn = jax.value_and_grad(logp_fn)
    logdensity, logdensity_grad = grad_fn(init_position)

    init_state = HMCState(
        init_position,
        logdensity,
        logdensity_grad,
    )

    # Calculate initial free energy
    delta_F_nominator_sum = jnp.exp(LAMBDAS / ZERO_TEMP * logdensity)
    delta_F_denominator_sum = 1
    delta_F = -jnp.log(delta_F_nominator_sum / delta_F_denominator_sum)

    delta_F = jnp.clip(
        delta_F,
        min=None,
        max=BIAS_CAP,
    )
    # Calculate initial bias value for storing
    V = jnp.exp(
        LAMBDAS / ZERO_TEMP * logdensity + delta_F
    )
    bias_value = -jnp.log(jnp.sum(V) / NUM_LAMBDA)
    # Set up initial BiasState
    init_bias_state = BiasState(
        init_state,
        delta_F_nominator_sum,
        delta_F_denominator_sum,
        delta_F,
        bias_value,
        0.0,
    )

    # Pure function to perform one step of the algorithm, takes a bias state
    # and rng_keys and returns a bias_state, position and info
    def _one_step(bias_state, xs):
        _, rng_key = xs
        # unpack, done for convenience only
        state = bias_state.state
        delta_F_nominator_sum = bias_state.delta_F_nominator_sum
        delta_F_denominator_sum = bias_state.delta_F_denominator_sum
        delta_F = bias_state.delta_F
        # partial function evaluation, so that bias arguments are always the
        # same
        bias_function = partial(
            bias_potential,
            delta_F=delta_F
        )

        def logp_biased(pos):
            # bias is subtracked instead of added, due to the different sign
            # of logp_fn in comparison to molecular dynamics (probability
            # distribution vs. potential energy)
            logp = logp_fn(pos)
            return logp - bias_function(-logp)

        grad_fn = jax.value_and_grad(logp_biased)
        logdensity, logdensity_grad = grad_fn(state.position)
        state = HMCState(
            position=state.position,
            logdensity=logdensity,
            logdensity_grad=logdensity_grad
        )
        biased_kernel = algorithm(
            logp_biased,
            step_size=STEPSIZE,
            inverse_mass_matrix=jnp.ones(2),
            num_integration_steps=INTEGRATION_STEPS,
            # Uncomment for toying with integrators
            # integrator=yoshida,
        ).step

        for _ in range(HMC_STEPS):
            # XXX update rng_key?
            rng_key, subkey = jax.random.split(rng_key)
            state, info = biased_kernel(subkey, state)

        # Prepare info and outputs
        position = state.position
        logdensity = state.logdensity
        potential = bias_potential(-logdensity, delta_F)
        stats = {
            "diverging": info.is_divergent,
            "energy": info.energy,
            # "tree_depth": info.num_trajectory_expansions,
            # "n_steps": info.num_integration_steps,
            "acceptance_rate": info.acceptance_rate,
            # logdensity we get from state contains the bias, so we correct for
            # it
            "lp": logdensity + potential,
            "bias_value": potential,
            "delta_F": bias_state.delta_F,
        }

        # Update bias
        new_delta_F_nominator_sum, new_delta_F_denominator_sum, new_delta_F = \
            update_delta_F(
                -logdensity,
                delta_F_nominator_sum,
                delta_F_denominator_sum,
                delta_F,
                potential,
            )

        new_bias_state = BiasState(
            state,
            new_delta_F_nominator_sum,
            new_delta_F_denominator_sum,
            new_delta_F,
            potential,
            bias_state.count + 1,
        )

        return new_bias_state, (position, stats)

    # This is kept mostly from the reference implementation
    progress_bar = adaptation_kwargs.pop("progress_bar", False)

    keys = jax.random.split(seed, draws)
    scan_fn = blackjax.progress_bar.gen_scan_fn(draws, progress_bar)
    _, (samples, stats) = scan_fn(
        _one_step,
        init_bias_state,        # This is changed from the reference
        (jnp.arange(draws), keys),
    )

    return samples, stats
