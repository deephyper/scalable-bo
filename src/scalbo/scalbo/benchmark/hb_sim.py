import time
from deephyper.problem import HpProblem
from deephyper.evaluator import profile

nb_dim = 1
domain = (-10.0, 10.0)
hp_problem = HpProblem()
hp_problem.add_hyperparameter(domain, f"a")

@profile
def run(config):

    optuna_trial_id = config.get("optuna_trial_id")
    x = config.get("resource", 10)
    if optuna_trial_id:
        time.sleep(1/4)
    else:
        time.sleep(10/4)
    a = config["a"]
    y = a * x 
    return y
