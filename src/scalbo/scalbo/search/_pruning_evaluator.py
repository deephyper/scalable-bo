import optuna

from deephyper.evaluator import SerialEvaluator, Job


class PruningEvaluator(SerialEvaluator):

    def __init__(self, run_function, num_workers: int = 1, callbacks: list = None, run_function_kwargs: dict = None, study: optuna.Study=None):
        super().__init__(run_function, num_workers, callbacks, run_function_kwargs)
        self._study = study
    
    def _on_launch(self, job: Job):
        super()._on_launch(job)

        trial_id = self._study._storage.create_new_trial(self._study._study_id)
        trial = optuna.Trial(self._study, trial_id)
        self.run_function_kwargs["optuna_trial"] = trial
    
    # def _on_done(self, job):
    #     super()._on_done(job)

    #     value = job.result
    #     step, pruned = job.other["step"], job.other["pruned"]

    #     trial = self._run_function_kwargs["optuna_trial"]

