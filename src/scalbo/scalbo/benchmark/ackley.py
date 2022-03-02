import numpy as np
from deephyper.problem import HpProblem

nb_dim = 10
problem = HpProblem()
for i in range(nb_dim):
    problem.add_hyperparameter((-32.768, 32.768), f"x{i}")

def ackley(x, a=20, b=0.2, c=2*np.pi):
    d = len(x)
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / d))
    term2 = -np.exp(s2 / d)
    y = term1 + term2 + a + np.exp(1)
    return y

def run(config):
    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -ackley(x)