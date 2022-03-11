import time
import numpy as np
from deephyper.problem import HpProblem
from deephyper.evaluator import profile

nb_dim = 6
domain = (0.0, 1.0)
hp_problem = HpProblem()
for i in range(nb_dim):
    hp_problem.add_hyperparameter(domain, f"x{i}")

def hartmann6D(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [[10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]]
    )
    P = 1e-4 * np.array(
        [[1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]]
    )
    X = np.array([x for _ in range(4)])
    inner = np.sum(np.multiply(A, np.square(X-P)), axis=1)
    outer = np.sum(alpha * np.exp(-inner))
    y = -(2.58 + outer) / 1.94
    return y

@profile
def run(config):
    t_sleep = np.random.normal(loc=60, scale=20)
    t_sleep = max(t_sleep, 0)
    time.sleep(t_sleep)
    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -hartmann6D(x)


if __name__ == "__main__":
    from scalbo.benchmark.hartmann6D import run

    config = {f"x{i}": 0 for i in range(6)}
    obj = run(config)
    print(obj["timestamp_end"] - obj["timestamp_start"])

    obj = run(config)
    print(obj["timestamp_end"] - obj["timestamp_start"])