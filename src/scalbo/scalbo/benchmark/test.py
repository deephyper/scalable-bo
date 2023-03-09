import time

from deephyper.problem import HpProblem
from deephyper.evaluator import profile

hp_problem = HpProblem()
hp_problem.add_hyperparameter((-10.0, 10.0), "x0")
hp_problem.add_hyperparameter((0, 5), "x1")
hp_problem.add_hyperparameter(list(range(100)), "x2")


@profile
def run(job, optuna_trial=None):

    config = job.parameters

    min_b, max_b = 1, 50
    const = config["x0"] + config["x1"] + config["x2"]
    objective_i = 0

    other_metadata = {}

    if optuna_trial:
        for budget_i in range(min_b, max_b + 1):
            time.sleep(0.25)
            objective_i += const
            optuna_trial.report(objective_i, step=budget_i)
            if optuna_trial.should_prune():
                break
        objective = objective_i
    else:
        for budget_i in range(min_b, max_b + 1):
            time.sleep(0.25)
            objective_i += const
            job.record(budget_i, objective_i)
            if job.stopped():
                break
        objective = objective_i

        if hasattr(job, "stopper") and hasattr(job.stopper, "infos_stopped"):
            other_metadata["infos_stopped"] = job.stopper.infos_stopped

    metadata = {
        "budget": budget_i,
        "stopped": budget_i < max_b,
    }
    metadata.update(other_metadata)
    return {
        "objective": objective,
        "metadata": metadata,
    }
