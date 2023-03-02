import numpy as np


from pybnn.lc_extrapolation.learning_curves import MCMCCurveModelCombination
from deephyper.stopper._stopper import Stopper


def area_learning_curve(z, f, z_max) -> float:
    assert len(z) == len(f)
    assert z[-1] <= z_max
    area = 0
    for i in range(1, len(z)):
        # z: is always monotinic increasing but not f!
        area += (z[i] - z[i - 1]) * f[i - 1]
    if z[-1] < z_max:
        area += (z_max - z[-1]) * f[-1]
    return area


class WPModelStopper(Stopper):
    """Weighted Probabilistic Model Stopper based on a learning curve model."""

    def __init__(
        self,
        max_steps: int,
        min_steps: int = 4,
        min_done_for_outlier_detection=10,
        iqr_factor_for_outlier_detection=1.5,
        prob_promotion=0.9,
        early_stopping_patience=0.25,
        objective_returned="last",
        random_state=None,
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.min_steps = min_steps

        self.min_obs_to_fit = min_steps

        self.min_done_for_outlier_detection = min_done_for_outlier_detection
        self.iqr_factor_for_outlier_detection = iqr_factor_for_outlier_detection

        self.prob_promotion = prob_promotion
        if type(early_stopping_patience) is int:
            self.early_stopping_patience = early_stopping_patience
        elif type(early_stopping_patience) is float:
            self.early_stopping_patience = int(early_stopping_patience * self.max_steps)
        else:
            raise ValueError("early_stopping_patience must be int or float")
        self.objective_returned = objective_returned

        self._rung = 0

        self._lc_objectives = []

    def _compute_halting_step(self):
        return self.min_steps * self.min_obs_to_fit**self._rung

    def _retrieve_best_objective(self) -> float:
        search_id, _ = self.job.id.split(".")
        objectives = []
        for obj in self.job.storage.load_out_from_all_jobs(search_id):
            try:
                objectives.append(float(obj))
            except ValueError:
                pass
        if len(objectives) > 0:
            return np.max(objectives)
        else:
            return np.max(self.observations[1])

    def _get_competiting_objectives(self, rung) -> list:
        search_id, _ = self.job.id.split(".")
        values = self.job.storage.load_metadata_from_all_jobs(
            search_id, f"completed_rung_{rung}"
        )
        values = [float(v) for v in values]
        return values

    def observe(self, budget: float, objective: float):
        super().observe(budget, objective)
        self._budget = self.observed_budgets[-1]
        self._lc_objectives.append(self.objective)
        self._objective = self._lc_objectives[-1]

        # For Early-Stopping based on Patience
        if (
            not (hasattr(self, "_local_best_objective"))
            or self._objective > self._local_best_objective
        ):
            self._local_best_objective = self._objective
            self._local_best_step = self.step

        halting_step = self._compute_halting_step()
        if self._budget >= halting_step:
            self.job.storage.store_job_metadata(
                self.job.id, f"completed_rung_{self._rung}", str(self._objective)
            )

    def stop(self) -> bool:

        # Enforce Pre-conditions Before Learning-Curve based Early Discarding
        if super().stop():
            print("Stopped after reaching the maximum number of steps.")
            self.infos_stopped = "max steps reached"
            return True

        if self.step - self._local_best_step >= self.early_stopping_patience:
            print(
                f"Stopped after reaching {self.early_stopping_patience} steps without improvement."
            )
            self.infos_stopped = "early stopping"
            return True

        # This condition will enforce the stopper to stop the evaluation at the first step
        # for the first evaluation (The FABOLAS method does the same, bias the first samples with
        # small budgets)
        self.best_objective = self._retrieve_best_objective()

        halting_step = self._compute_halting_step()
        if self.step < max(self.min_steps, self.min_obs_to_fit):

            if self.step >= halting_step:
                competing_objectives = self._get_competiting_objectives(self._rung)
                if len(competing_objectives) > self.min_done_for_outlier_detection:
                    q1 = np.quantile(
                        competing_objectives,
                        q=0.25,
                    )
                    q3 = np.quantile(
                        competing_objectives,
                        q=0.75,
                    )
                    iqr = q3 - q1
                    # lower than the minimum of a box plot
                    if (
                        self._objective
                        < q1 - self.iqr_factor_for_outlier_detection * iqr
                    ):
                        print(
                            f"Stopped early because of abnormally low objective: {self._objective}"
                        )
                        self.infos_stopped = "outlier"
                        return True
                self._rung += 1

            return False

        # Check if the halting budget condition is met
        if self.step < halting_step:
            return False

        # Check if the evaluation should be stopped based on LC-Model

        # Fit and predict the performance of the learning curve model
        z_train = self.observed_budgets
        y_train = self._lc_objectives
        z_train, y_train = np.asarray(z_train), np.negative(y_train)
        self.lc_model = MCMCCurveModelCombination(
            xlim=self.max_steps + 1,
            recency_weighting=False,
            nwalkers=1,
            burn_in=200,
            nsamples=1_000,
        )
        try:
            self.lc_model.fit(z_train, y_train)
        except ValueError:  # Failed the fitting
            self._rung += 1
            return False

        # Check if the configuration is promotable based on its predicted objective value
        # pred = self.lc_model.predictive_distribution(self.max_steps)
        # p = np.mean(pred <= self.best_objective)
        try:
            p = self.lc_model.posterior_prob_x_greater_than(
                x=self.max_steps, y=-self.best_objective
            )
        except AttributeError:  # Failed the fitting
            self._rung += 1
            return False

        # Return whether the configuration should be stopped
        if p <= self.prob_promotion:
            self._rung += 1
        else:
            print(
                f"Stopped because the probability of performing worse is {p} > {self.prob_promotion}"
            )
            self.infos_stopped = f"prob={p:.3f}"

            return True

    @property
    def objective(self):
        if self.objective_returned == "last":
            return self.observations[-1][-1]
        elif self.objective_returned == "best":
            return max(self.observations[-1])
        elif self.objective_returned == "alc":
            z, y = self.observations
            return area_learning_curve(z, y, z_max=self.max_steps)
        else:
            raise ValueError("objective_returned must be one of 'last', 'best', 'alc'")
