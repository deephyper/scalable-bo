import time

from deephyper.problem import HpProblem
from deephyper.evaluator import profile

hp_problem = HpProblem()
hp_problem.add_hyperparameter((-10.0, 10.0), "x0")
hp_problem.add_hyperparameter((0, 5), "x1")
hp_problem.add_hyperparameter(list(range(100)), "x2")


@profile
def run(config, optuna_trial=None):
    max_steps = 50
    const = config["x0"] + config["x1"] + config["x2"]
    score = 0
    pruned = False
    for step in range(1, max_steps + 1):

        time.sleep(0.25)
        score += const

        if optuna_trial:
            optuna_trial.report(score, step=step)

            # Prune trial if needed
            if optuna_trial.should_prune():
                pruned = True
                break

    if optuna_trial:
        return {"objective": score, "pruned": pruned, "budget": step}
    else:
        return score
