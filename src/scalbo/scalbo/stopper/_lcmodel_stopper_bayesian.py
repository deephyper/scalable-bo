import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from deephyper.stopper._stopper import Stopper


# Budget allocation models
def b_lin2(z, nu=[1, 1]):
    return nu[1] * (z - 1) + nu[0]


def b_exp2(z, nu=[1, 2]):
    return nu[0] * jnp.power(nu[1], z - 1)


# Learning curves models
def f_lin2(z, b, rho):
    return rho[1] * b(z) + rho[0]


def f_loglin2(z, b, rho):
    Z = jnp.log(b(z))
    Y = rho[1] * Z + rho[0]
    y = jnp.exp(Y)
    return y  # !maximization


def f_loglin3(z, b, rho):
    Z = jnp.log(b(z))
    Y = rho[2] * jnp.power(Z, 2) + rho[1] * Z + rho[0]
    y = jnp.exp(Y)
    return y  # !maximization


def f_loglin4(z, b, rho):
    Z = jnp.log(b(z))
    Y = rho[3] * jnp.power(Z, 3) + rho[2] * jnp.power(Z, 2) + rho[1] * Z + rho[0]
    y = jnp.exp(Y)
    return y  # !maximization


def f_pow3(z, b, rho):
    return rho[0] - rho[1] * b(z) ** -rho[2]


def f_mmf4(z, b, rho):
    return (rho[0] * rho[1] + rho[2] * jnp.power(b(z), rho[3])) / (
        rho[1] + jnp.power(b(z), rho[3])
    )


# Utility to estimate parameters of learning curve model
# The combination of "partial" and "static_argnums" is necessary
# with the "f" lambda function passed as argument
@partial(jax.jit, static_argnums=(1,))
def residual_least_square(rho, f, z, y):
    """Residual for least squares."""
    return f(z, rho) - y


def prob_model(z=None, y=None, f=None, rho_mu_prior=None, num_obs=None):
    rho_mu_prior = jnp.array(rho_mu_prior)
    rho_sigma_prior = 1.0
    rho = numpyro.sample("rho", dist.Normal(rho_mu_prior, rho_sigma_prior))
    # sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    sigma = 0.01
    mu = f(z[:num_obs], rho)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y[:num_obs])


@partial(jax.jit, static_argnums=(0,))
def predict_moments_from_posterior(f, X, posterior_samples):
    vf_model = jax.vmap(f, in_axes=(None, 0))
    posterior_mu = vf_model(X, posterior_samples)
    mean_mu = jnp.mean(posterior_mu, axis=0)
    std_mu = jnp.std(posterior_mu, axis=0)
    return mean_mu, std_mu


class BayesianLearningCurveRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        f_model=f_loglin3,
        f_model_num_params=3,
        b_model=b_lin2,
        max_trials_ls_fit=5,
        mcmc_num_warmup=200,
        mcmc_num_samples=200,
        n_jobs=-1,
        random_state=None,
        verbose=0,
        batch_size=100,
    ):
        self.b_model = b_model
        self.f_model = lambda z, rho: f_model(z, self.b_model, rho)
        self.f_nparams = f_model_num_params
        self.mcmc_num_warmup = mcmc_num_warmup
        self.mcmc_num_samples = mcmc_num_samples
        self.max_trials_ls_fit = max_trials_ls_fit
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.rho_mu_prior_ = np.zeros((self.f_nparams,))

        self.batch_size = batch_size
        self.X_ = np.zeros((self.batch_size,))
        self.y_ = np.zeros((self.batch_size,))

    def fit(self, X, y, update_prior=True):

        check_X_y(X, y, ensure_2d=False)

        # !Trick for performance to avoid performign JIT again and again
        # !This will fix the shape of inputs of the model for numpyro
        # !see https://github.com/pyro-ppl/numpyro/issues/441
        num_samples = len(X)
        assert num_samples <= self.batch_size
        self.X_[:num_samples] = X[:]
        self.y_[:num_samples] = y[:]

        if update_prior:
            self.rho_mu_prior_[:] = self._fit_learning_curve_model_least_square(X, y)[:]

        if not (hasattr(self, "kernel_")):
            self.kernel_ = NUTS(
                model=lambda z, y, rho_mu_prior: prob_model(
                    z, y, self.f_model, rho_mu_prior, num_obs=num_samples
                ),
            )

            self.mcmc_ = MCMC(
                self.kernel_,
                num_warmup=self.mcmc_num_warmup,
                num_samples=self.mcmc_num_samples,
                progress_bar=self.verbose,
                jit_model_args=True,
            )

        seed = self.random_state.randint(low=0, high=2**32)
        rng_key = jax.random.PRNGKey(seed)
        self.mcmc_.run(rng_key, z=self.X_, y=self.y_, rho_mu_prior=self.rho_mu_prior_)

        if self.verbose:
            self.mcmc_.print_summary()

        return self

    def predict(self, X, return_std=True):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X, ensure_2d=False)

        posterior_samples = self.mcmc_.get_samples()

        mean_mu, std_mu = predict_moments_from_posterior(
            self.f_model, X, posterior_samples["rho"]
        )

        if return_std:
            return mean_mu, std_mu

        return mean_mu

    def _fit_learning_curve_model_least_square(
        self,
        z_train,
        y_train,
    ):
        """The learning curve model is assumed to be modeled by 'f' with
        interface f(z, rho).
        """

        seed = self.random_state.randint(low=0, high=2**32)
        random_state = check_random_state(seed)

        z_train = np.asarray(z_train)
        y_train = np.asarray(y_train)

        # compute the jacobian
        # using the true jacobian is important to avoid problems
        # with numerical errors and approximations! indeed the scale matters
        # a lot when approximating with finite differences
        def fun_wrapper(rho, f, z, y):
            return np.array(residual_least_square(rho, f, z, y))

        if not (hasattr(self, "jac_residual_ls_")):
            self.jac_residual_ls_ = partial(jax.jit, static_argnums=(1,))(
                jax.jacfwd(residual_least_square, argnums=0)
            )

        def jac_wrapper(rho, f, z, y):
            return np.array(self.jac_residual_ls_(rho, f, z, y))

        results = []
        mse_hist = []

        for _ in range(self.max_trials_ls_fit):

            rho_init = random_state.randn(self.f_nparams)

            try:
                res_lsq = least_squares(
                    fun_wrapper,
                    rho_init,
                    args=(self.f_model, z_train, y_train),
                    method="lm",
                    jac=jac_wrapper,
                )
            except ValueError:
                continue

            mse_res_lsq = np.mean(res_lsq.fun**2)
            mse_hist.append(mse_res_lsq)
            results.append(res_lsq.x)

        i_best = np.nanargmin(mse_hist)
        res = results[i_best]
        return res


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


class LCModelStopper(Stopper):
    """Stopper based on a learning curve model."""

    def __init__(
        self,
        max_steps: int,
        min_steps: int = 1,
        lc_model="pow3",
        kappa=1.96,
        random_state=None,
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.min_steps = min_steps

        lc_model = "f_" + lc_model
        lc_model_num_params = int(lc_model[-1])
        lc_model = getattr(sys.modules[__name__], lc_model)

        self.kappa = kappa

        self.min_obs_to_fit = lc_model_num_params

        self._rung = 0

        # compute the step at which to stop based on steps allocation policy
        max_rung = np.floor(
            np.log(self.max_steps / self.min_steps) / np.log(self.min_obs_to_fit)
        )
        self.max_steps_ = int(self.min_steps * self.min_obs_to_fit**max_rung)
        self._step_max_utility = self.max_steps_

        self.lc_model = BayesianLearningCurveRegressor(
            f_model=lc_model,
            f_model_num_params=lc_model_num_params,
            random_state=random_state,
            batch_size=self.max_steps_,
        )

    def _compute_halting_step(self):
        return self.min_steps * self.min_obs_to_fit**self._rung

    def _fit_and_predict_lc_model_performance(self):
        """Estimate the LC Model and Predict the performance at b.

        Returns:
            (step, objective): a tuple of (step, objective) at which the estimation was made.
        """

        # By default (no utility used) predict at the last possible step.
        z_pred = self.max_steps
        z_opt = self.max_steps_

        z_train, y_train = self.observations
        z_train, y_train = np.asarray(z_train), np.asarray(y_train)

        self.lc_model.fit(z_train, y_train)

        mean_pred, std_pred = self.lc_model.predict([z_pred])
        ucb_pred = mean_pred[0] + self.kappa * std_pred[0]

        return z_opt, ucb_pred

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
            return None

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
        self._objective = self.observed_objectives[-1]

        halting_step = self._compute_halting_step()
        if self._budget >= halting_step:
            self.job.storage.store_job_metadata(
                self.job.id, f"completed_rung_{self._rung}", str(self._objective)
            )

    def stop(self) -> bool:

        if not (hasattr(self, "best_objective")):
            # print("START")
            self.best_objective = self._retrieve_best_objective()

        # Enforce Pre-conditions Before Learning-Curve based Early Discarding
        if super().stop():
            return True

        # This condition will enforce the stopper to stop the evaluation at the first step
        # for the first evaluation (The FABOLAS method does the same, bias the first samples with
        # small budgets)
        if self.best_objective is None:
            return True

        halting_step = self._compute_halting_step()
        if self.step < max(self.min_steps, self.min_obs_to_fit):

            if self.step >= halting_step:
                # TODO: make fixed parameter accessible
                competing_objectives = self._get_competiting_objectives(self._rung)
                if len(competing_objectives) > 10:
                    q_objective = np.quantile(competing_objectives, q=0.33)
                    if self._objective < q_objective:
                        return True
                self._rung += 1

            return False

        # Check if the halting budget condition is met
        if self.step < halting_step and self.step < self._step_max_utility:
            return False

        # Check if the evaluation should be stopped based on LC-Model

        # Fit and predict the performance of the learning curve model
        z_opt, y_pred = self._fit_and_predict_lc_model_performance()
        # print(f"{z_opt=} - {y_pred=} - best={self.best_objective}")

        # Check if the configuration is promotable based on its predicted objective value
        promotable = (self.best_objective is None or y_pred > self.best_objective) and (
            self.step < z_opt
        )

        # Return whether the configuration should be stopped
        if promotable:
            self._rung += 1

            if self.step >= self.max_steps_:
                return True

        return not (promotable)
