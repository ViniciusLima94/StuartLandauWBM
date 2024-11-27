import jax


def randn(mu: float = 0, sig: float = 0, size: tuple = (1,), seed: int = 0):

    return jax.random.normal(jax.random.key(seed), shape=size)
