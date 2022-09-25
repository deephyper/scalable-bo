import copy
import logging
import time

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import optuna
from deephyper.search._search import Search


def convert_to_optuna_distribution(cs_hp):
    name = cs_hp.name
    if isinstance(cs_hp, csh.UniformIntegerHyperparameter):
        if cs_hp.log:
            dist = optuna.distributions.IntLogUniformDistribution(
                low=cs_hp.lower, high=cs_hp.upper
            )
        else:
            dist = optuna.distributions.IntUniformDistribution(
                low=cs_hp.lower, high=cs_hp.upper
            )
    elif isinstance(cs_hp, csh.UniformFloatHyperparameter):
        if cs_hp.log:
            dist = optuna.distributions.LogUniformDistribution(
                low=cs_hp.lower, high=cs_hp.upper
            )
        else:
            dist = optuna.distributions.UniformDistribution(
                low=cs_hp.lower, high=cs_hp.upper
            )
    elif isinstance(cs_hp, csh.CategoricalHyperparameter):
        dist = optuna.distributions.CategoricalDistribution(choices=cs_hp.choices)
    elif isinstance(cs_hp, csh.OrdinalHyperparameter):
        dist = optuna.distributions.CategoricalDistribution(choices=cs_hp.sequence)
    else:
        raise TypeError(f"Cannot convert hyperparameter of type {type(cs_hp)}")

    return name, dist


def convert_to_optuna_space(cs_space):
    # verify pre-conditions
    if not (isinstance(cs_space, cs.ConfigurationSpace)):
        raise TypeError("Input space should be of type ConfigurationSpace")

    if len(cs_space.get_conditions()) > 0:
        raise RuntimeError("Cannot convert a ConfigSpace with Conditions!")

    if len(cs_space.get_forbiddens()) > 0:
        raise RuntimeError("Cannot convert a ConfigSpace with Forbiddens!")

    # convert the ConfigSpace to deephyper.skopt.space.Space
    distributions = {}
    for cs_hp in cs_space.get_hyperparameters():
        name, dist = convert_to_optuna_distribution(cs_hp)
        distributions[name] = dist

    return distributions


class HB(Search):
    """HB"""

    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        n_initial_points=10,
        initial_points=None,
        sync_communication: bool = False,
        **kwargs,
    ):

        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        self._n_initial_points = n_initial_points
        self._initial_points = []
        if initial_points is not None and len(initial_points) > 0:
            for point in initial_points:
                if isinstance(point, list):
                    self._initial_points.append(point)
                elif isinstance(point, dict):
                    self._initial_points.append(
                        [point[hp_name] for hp_name in problem.hyperparameter_names]
                    )
                else:
                    raise ValueError(
                        f"Initial points should be dict or list but {type(point)} was given!"
                    )

        # check if it is possible to convert the ConfigSpace to standard skopt Space
        self._min_resource = 1
        self._max_resource = 10
        self._opt_space = convert_to_optuna_space(self._problem.space)
        self._opt_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=random_state),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=self._min_resource,
                max_resource=self._max_resource,
                reduction_factor=3,
            ),
        )
        self._running_trials = {}
        self._pending_trials = {}

        self._gather_type = "ALL" if sync_communication else "BATCH"

    def _ask_v1(self, n_points):
        configs = []
        pending_trials = list(self._pending_trials.values())
        for i in range(n_points):
            if i < len(pending_trials):
                trial = self._pending_trials.pop(pending_trials[i].number)
            else:
                trial = self._opt_study.ask(self._opt_space)
            self._running_trials[trial.number] = trial
            logging.info(f"ask: #{trial.number}")
            
            if "resource" in trial.user_attrs:
                trial.set_user_attr("resource", trial.user_attrs["resource"]+1)
            else:
                trial.set_user_attr("resource", self._min_resource)

            config = {p: trial.params[p] for p in trial.params}
            config["trial_id"] = trial.number
            config["resource"] = trial.user_attrs["resource"]
            configs.append(config)
            logging.info(f"{config=}")

        return configs

    def _ask(self, n_points):
        configs = []
        for i in range(n_points):
            trial = self._opt_study.ask(self._opt_space)
            config = {p: trial.params[p] for p in trial.params}
            config["optuna_trial"] = trial
            configs.append(config)
        return configs
    
    def _tell(self, trials, y_batch):

        for trial, y in zip(trials, y_batch):

            trial.report(y["objective"], step=y["step"])
            if y["pruned"]:
                self._opt_study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                self._opt_study.tell(trial, y["objective"])

    def _search(self, max_evals, timeout):

        num_evals_done = 0

        logging.info(f"Asking {self._evaluator.num_workers} initial configurations...")
        t1 = time.time()
        new_batch = self._ask(n_points=self._evaluator.num_workers)
        logging.info(f"Asking took {time.time() - t1:.4f} sec.")

        # submit new configurations
        logging.info(f"Submitting {len(new_batch)} configurations...")
        t1 = time.time()
        self._evaluator.submit(new_batch)
        logging.info(f"Submition took {time.time() - t1:.4f} sec.")

        # Main loop
        while max_evals < 0 or num_evals_done < max_evals:
            # Collecting finished evaluations
            logging.info("Gathering jobs...")
            t1 = time.time()
            new_results = self._evaluator.gather(self._gather_type, size=1)
            logging.info(
                f"Gathered {len(new_results)} job(s) in {time.time() - t1:.4f} sec."
            )

            if len(new_results) > 0:

                logging.info("Dumping evaluations...")
                t1 = time.time()
                self._evaluator.dump_evals(log_dir=self._log_dir)
                logging.info(f"Dumping took {time.time() - t1:.4f} sec.")

                num_received = len(new_results)
                num_evals_done += num_received

                if max_evals > 0 and num_evals_done >= max_evals:
                    break

                # Transform configurations to list to fit optimizer
                logging.info("Transforming received configurations to list...")
                t1 = time.time()

                trials = []
                opt_y = []
                for job_i in new_results:
                    cfg = copy.deepcopy(job_i.config)
                    cfg.pop("job_id")

                    obj = {"objective": job_i.result, "step": job_i.other["step"], "pruned": job_i.other["pruned"]}

                    trials.append(cfg["optuna_trial"])
                    opt_y.append(obj)

                logging.info(f"Transformation took {time.time() - t1:.4f} sec.")

                logging.info("Fitting the optimizer...")
                t1 = time.time()

                if len(opt_y) > 0:
                    self._tell(trials, opt_y)
                    logging.info(f"Fitting took {time.time() - t1:.4f} sec.")

                logging.info(f"Asking {len(new_results)} new configurations...")
                t1 = time.time()
                new_batch = self._ask(n_points=len(new_results))
                logging.info(f"Asking took {time.time() - t1:.4f} sec.")

                # submit new configurations
                logging.info(f"Submitting {len(new_batch)} configurations...")
                t1 = time.time()
                self._evaluator.submit(new_batch)
                logging.info(f"Submition took {time.time() - t1:.4f} sec.")
