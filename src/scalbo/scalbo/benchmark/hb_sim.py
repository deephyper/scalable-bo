import time
from deephyper.problem import HpProblem
from deephyper.evaluator import profile

nb_dim = 1
domain = (-10.0, 10.0)
hp_problem = HpProblem()
hp_problem.add_hyperparameter(domain, f"a")

@profile
def run(config):

    max_x = 10
    optuna_trial = config.get("optuna_trial")
    a = config["a"]
    for x in range(1, max_x+1):
        time.sleep(1/4)
        y = a * x 
        if optuna_trial:
            optuna_trial.report(y, step=x)
            if optuna_trial.should_prune():
                return {"pruned": True, "objective": y, "step": x}
    if optuna_trial:
        return {"pruned": False, "objective": y, "step": x}
    else:
        return y
