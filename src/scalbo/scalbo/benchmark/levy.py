import time
import numpy as np
from deephyper.problem import HpProblem
from deephyper.evaluator import profile

nb_dim = 5
domain = (-10.0, 10.0)
hp_problem = HpProblem()
for i in range(nb_dim):
    hp_problem.add_hyperparameter(domain, f"x{i}")

def levy(x):
    z = 1 + (x - 1) / 4
    return (
        np.sin(np.pi * z[0]) ** 2
        + np.sum((z[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * z[:-1] + 1) ** 2))
        + (z[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * z[-1]) ** 2)
    )

@profile
def run(config):
    t_sleep = np.random.normal(loc=60, scale=20)
    t_sleep = max(t_sleep, 0)
    time.sleep(t_sleep)
    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -levy(x)


if __name__ == "__main__":
    config = {f"x{i}": 1 for i in range(5)}
    obj = run(config)
    print(f"{obj=}")