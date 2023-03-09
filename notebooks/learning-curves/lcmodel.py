import jax.numpy as jnp


# Budget allocation models
def b_lin2(z, nu=[1, 1]):
    return nu[1] * (z-1) + nu[0]


def b_exp2(z, nu=[1, 2]):
    return nu[0] * jnp.power(nu[1], z-1)


# Learning curves models
def f_lin2(z, b, rho):
    return rho[1] * b(z) + rho[0]


def f_loglin2(z, b, rho):
    Z = jnp.log(z)
    Y = rho[1] * Z + rho[0]
    y = jnp.exp(Y)
    return y


def f_loglin3(z, b, rho):
    Z = jnp.log(z)
    Y = rho[2] * jnp.power(Z, 2) + rho[1] * Z + rho[0]
    y = jnp.exp(Y)
    return y


def f_loglin4(z, b, rho):
    Z = jnp.log(z)
    Y = rho[3] * jnp.power(Z, 3) + rho[2] * jnp.power(Z, 2) + rho[1] * Z + rho[0]
    y = jnp.exp(Y)
    return y


def f_pow3(z, b, rho):
    return rho[0] - rho[1] * b(z)**-rho[2]


def f_mmf4(z, b, rho):
    return (rho[0] * rho[1] + rho[2] * jnp.power(b(z), rho[3])) / (rho[1] + jnp.power(b(z), rho[3]))