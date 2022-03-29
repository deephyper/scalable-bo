import time
import numpy as np
from deephyper.problem import HpProblem
from deephyper.evaluator import profile

nb_dim = 10
domain = (-600.0, 600.0)
hp_problem = HpProblem()
for i in range(nb_dim):
    hp_problem.add_hyperparameter(domain, f"x{i}")

def griewank(x, fr=4000):
    n = len(x)
    j = np.arange(1.0, n + 1)
    s = np.sum(x ** 2)
    p = np.prod(np.cos(x / np.sqrt(j)))
    return s / fr - p + 1

@profile
def run(config):
    t_sleep = np.random.normal(loc=60, scale=20)
    t_sleep = max(t_sleep, 0)
    time.sleep(t_sleep)
    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -griewank(x)


if __name__ == "__main__":
    config = {f"x{i}": 0 for i in range(5)}
    obj = run(config)
    print(f"{obj=}")