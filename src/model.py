import pymc as pm


def energy_function(x):
    return (
        1.34549 * x[0] * x[0] * x[0] * x[0]
        + 1.90211 * x[0] * x[0] * x[0] * x[1]
        + 3.92705 * x[0] * x[0] * x[1] * x[1]
        - 6.44246 * x[0] * x[0]
        - 1.90211 * x[0] * x[1] * x[1] * x[1]
        + 5.58721 * x[0] * x[1]
        + 1.33481 * x[0]
        + 1.34549 * x[1] * x[1] * x[1] * x[1]
        - 5.55754 * x[1] * x[1]
        + 0.904586 * x[1]
        + 18.5598
    )


# def energy_function(x):
#     return (x[0]-0.14)**2 + (x[1] - 1)**2


with pm.Model() as myModel:
    # Just use a flat prior
    x = pm.Flat(
        'x',
        size=2,
    )

    distribution = pm.Potential(
        'distribution',
        # negative energy, as PyMC is designed to work with probability
        # distributions
        -energy_function(x),
    )
