from functools import partial

import jax
import jax.numpy as jnp
import blackjax

from typing import NamedTuple
from blackjax.mcmc.hmc import HMCState

# Use this for toying with other integrators
# from blackjax.mcmc.integrators import yoshida

NUM_LAMBDA = 121
SIGMA = 0.05
INTEGRATION_STEPS = 10
STEPSIZE = 0.01
HMC_STEPS = 10


class BiasState(NamedTuple):
    state: HMCState
    gaussian_centers: jax.Array
    delta_F_nominator_sum: jax.Array
    delta_F_denominator_sum: jax.Array
    delta_F: jax.Array
    bias_value: float
    count: float


def bias_potential(x, gaussian_centers, delta_F):
    # Reference implementation with for-loop
    # sum_for_V = 0
    # for it in range(NUM_LAMBDA):
    #     sum_for_V += jnp.exp(
    #         -(x - gaussian_centers[it]) ** 2 / (2 * SIGMA ** 2)
    #         + delta_F[it]
    #     )

    # return -jnp.log(sum_for_V / NUM_LAMBDA)
    V = jnp.exp(
        -(x - gaussian_centers) ** 2 / (2 * SIGMA ** 2)
        + delta_F
    )
    return -jnp.log(jnp.sum(V) / NUM_LAMBDA)


def update_delta_F(
    x,
    gaussian_centers,
    delta_F_nominator_sum,
    delta_F_denominator_sum,
    delta_F,
    potential,
):
    delta_F_nominator_sum += jnp.exp(
        (-(x - gaussian_centers) ** 2 / (2 * SIGMA ** 2))
        + potential * jnp.ones(NUM_LAMBDA)
    )

    delta_F_denominator_sum += jnp.exp(potential) * jnp.ones(NUM_LAMBDA)

    delta_F = -jnp.log(
        delta_F_nominator_sum / delta_F_denominator_sum
    )

    delta_F = jnp.clip(
        delta_F,
        min=None,
        max=5,
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
    last_state = init_state

    # Set up gaussians
    gaussian_centers = -3 + 6 * jnp.arange(NUM_LAMBDA) / (NUM_LAMBDA - 1)
    # Calculate initial bias amplitudes
    delta_F_nominator_sum = jnp.exp(
        -(last_state.position[0][0] - gaussian_centers) ** 2
        / (2 * SIGMA ** 2)
    )
    # Test starting with no bias
    # delta_F_nominator_sum = 1e-12 * jnp.ones(NUM_LAMBDA)
    delta_F_denominator_sum = jnp.ones(NUM_LAMBDA)
    delta_F = -jnp.log(delta_F_nominator_sum / delta_F_denominator_sum)

    delta_F = jnp.clip(
        delta_F,
        min=None,
        max=15,
    )
    # Calculate initial bias value for storing
    V = jnp.exp(
        -(last_state.position[0][0] - gaussian_centers) ** 2 / (2 * SIGMA ** 2)
        + delta_F
    )

    bias_value = -jnp.log(jnp.mean(V))
    # Set up initial BiasState
    init_bias_state = BiasState(
        last_state,
        gaussian_centers,
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
        gaussian_centers = bias_state.gaussian_centers
        delta_F_nominator_sum = bias_state.delta_F_nominator_sum
        delta_F_denominator_sum = bias_state.delta_F_denominator_sum
        delta_F = bias_state.delta_F
        # partial function evaluation, so that bias arguments are always the
        # same
        bias_function = partial(
            bias_potential,
            gaussian_centers=gaussian_centers,
            delta_F=delta_F
        )

        def logp_biased(pos):
            # bias is subtracked instead of added, due to the different sign
            # of logp_fn in comparison to molecular dynamics (probability
            # distribution vs. potential energy)
            return logp_fn(pos) - bias_function(pos[0][0])

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
            state, info = biased_kernel(rng_key, state)

        # Prepare info and outputs
        position = state.position
        potential = bias_potential(position[0][0], gaussian_centers, delta_F)

        stats = {
            "diverging": info.is_divergent,
            "energy": info.energy,
            # "tree_depth": info.num_trajectory_expansions,
            # "n_steps": info.num_integration_steps,
            "acceptance_rate": info.acceptance_rate,
            "lp": state.logdensity,
            "bias_value": potential,
            "delta_F": bias_state.delta_F,
        }

        # Update bias
        new_delta_F_nominator_sum, new_delta_F_denominator_sum, new_delta_F = \
            update_delta_F(
                position[0][0],
                gaussian_centers,
                delta_F_nominator_sum,
                delta_F_denominator_sum,
                delta_F,
                potential,
            )

        new_bias_state = BiasState(
            state,
            gaussian_centers,
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
