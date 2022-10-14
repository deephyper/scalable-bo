from ._optuna import execute_optuna

def execute(
    problem,
    sync,
    acq_func,
    strategy,
    timeout,
    random_state,
    log_dir,
    cache_dir,
    n_jobs,
    model,
):
    """Execute the TPE algorithm with HB.
    """
    execute_optuna(problem, timeout, random_state, log_dir, cache_dir, method="TPEHB")