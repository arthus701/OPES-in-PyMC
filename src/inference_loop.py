from functools import partial

import jax
import jax.numpy as jnp
import blackjax

# from blackjax.adaptation.base import get_filter_adapt_info_fn

from typing import NamedTuple
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.integrators import yoshida
# from blackjax.types import ArrayTree

NUM_LAMBDA = 11
SIGMA = 0.5
INTEGRATION_STEPS = 10
STEPSIZE = 0.1
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
        max=15,
    )

    return delta_F_nominator_sum, delta_F_denominator_sum, delta_F


def inference_loop(
    seed, init_position, logp_fn, draws, tune, target_accept,
    **adaptation_kwargs
):
    # algorithm_name = adaptation_kwargs.pop("algorithm", "nuts")
    # if algorithm_name == "nuts":
    #     algorithm = blackjax.nuts
    # elif algorithm_name == "hmc":
    #     algorithm = blackjax.hmc
    # else:
    #     raise ValueError(
    #         "Only supporting 'nuts' or 'hmc' as algorithm to draw samples."
    #     )
    adaptation_kwargs.pop("algorithm", "nuts")
    algorithm = blackjax.hmc

    # adapt = blackjax.window_adaptation(
    #     algorithm=algorithm,
    #     logdensity_fn=logp_fn,
    #     target_acceptance_rate=target_accept,
    #     num_integration_steps=10,
    #     adaptation_info_fn=get_filter_adapt_info_fn(),
    #     **adaptation_kwargs,
    # )
    # (last_state, tuned_params), _ = adapt.run(
    #     seed,
    #     init_position,
    #     num_steps=tune,
    # )

    grad_fn = jax.value_and_grad(logp_fn)
    logdensity, logdensity_grad = grad_fn(init_position)

    init_state = HMCState(
        init_position,
        logdensity,
        logdensity_grad,
    )
    last_state = init_state

    gaussian_centers = -3 + 6 * jnp.arange(NUM_LAMBDA) / (NUM_LAMBDA - 1)
    delta_F_nominator_sum = jnp.exp(
        -(last_state.position[0][0] - gaussian_centers) ** 2
        / (2 * SIGMA ** 2)
    )
    # delta_F_nominator_sum = 1e-12 * jnp.ones(NUM_LAMBDA)
    delta_F_denominator_sum = jnp.ones(NUM_LAMBDA)
    delta_F = -jnp.log(delta_F_nominator_sum / delta_F_denominator_sum)

    delta_F = jnp.clip(
        delta_F,
        min=None,
        max=15,
    )

    V = jnp.exp(
        -(last_state.position[0][0] - gaussian_centers) ** 2 / (2 * SIGMA ** 2)
        + delta_F
    )

    bias_value = -jnp.log(jnp.mean(V))

    init_bias_state = BiasState(
        last_state,
        gaussian_centers,
        delta_F_nominator_sum,
        delta_F_denominator_sum,
        delta_F,
        bias_value,
        0.0,
    )

    def _one_step(bias_state, xs):
        _, rng_key = xs
        # unpack
        state = bias_state.state
        gaussian_centers = bias_state.gaussian_centers
        delta_F_nominator_sum = bias_state.delta_F_nominator_sum
        delta_F_denominator_sum = bias_state.delta_F_denominator_sum
        delta_F = bias_state.delta_F

        bias_function = partial(
            bias_potential,
            gaussian_centers=gaussian_centers,
            delta_F=delta_F
        )

        def logp_biased(pos):
            # return logp_fn(pos) - bias_function(pos)
            return logp_fn(pos) - bias_function(pos[0][0])

        # print(bias_value_fn(bias_state))

        # biased_kernel = algorithm(logp_biased, **tuned_params).step
        biased_kernel = algorithm(
            logp_biased,
            step_size=STEPSIZE,
            inverse_mass_matrix=jnp.ones(2),
            num_integration_steps=INTEGRATION_STEPS,
            # integrator=yoshida,
        ).step
        for _ in range(HMC_STEPS):
            # XXX update rng_key?
            state, info = biased_kernel(rng_key, state)

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

    progress_bar = adaptation_kwargs.pop("progress_bar", False)

    keys = jax.random.split(seed, draws)
    scan_fn = blackjax.progress_bar.gen_scan_fn(draws, progress_bar)
    _, (samples, stats) = scan_fn(
        _one_step,
        init_bias_state,
        (jnp.arange(draws), keys),
    )

    return samples, stats
