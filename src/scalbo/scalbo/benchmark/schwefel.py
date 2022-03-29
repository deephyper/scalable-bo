import time
import numpy as np
from deephyper.problem import HpProblem
from deephyper.evaluator import profile

nb_dim = 10
domain = (-500.0, 500.0)
hp_problem = HpProblem()
for i in range(nb_dim):
    hp_problem.add_hyperparameter(domain, f"x{i}")

def schwefel(x):  # schw.m
    n = len(x)
    return 418.9829 * n - sum(x * np.sin(np.sqrt(np.abs(x))))

@profile
def run(config):
    t_sleep = np.random.normal(loc=60, scale=20)
    t_sleep = max(t_sleep, 0)
    time.sleep(t_sleep)
    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -schwefel(x)


if __name__ == "__main__":
    config = {f"x{i}": 420.9687 for i in range(5)}
    obj = run(config)
    print(f"{obj=}")