import numpy as np

from deephyper.problem import HpProblem
from deephyper.evaluator import profile

hp_problem = HpProblem()

# Model hyperparameters
ACTIVATIONS = [
    "elu",
    "gelu",
    "hard_sigmoid",
    "linear",
    "relu",
    "selu",
    "sigmoid",
    "softplus",
    "softsign",
    "swish",
    "tanh",
]
default_dense = [1000, 1000, 1000]
default_dense_feature_layers = [1000, 1000, 1000]

for i in range(len(default_dense)):

    hp_problem.add_hyperparameter(
        (10, 1024, "log-uniform"),
        f"dense_{i}",
        default_value=default_dense[i],
    )

    hp_problem.add_hyperparameter(
        (10, 1024, "log-uniform"),
        f"dense_feature_layers_{i}",
        default_value=default_dense_feature_layers[i],
    )

hp_problem.add_hyperparameter(ACTIVATIONS, "activation", default_value="relu")

# Optimization hyperparameters
hp_problem.add_hyperparameter(
    [
        "sgd",
        "rmsprop",
        "adagrad",
        "adadelta",
        "adam",
    ],
    "optimizer",
    default_value="sgd",
)

hp_problem.add_hyperparameter((0, 0.5), "dropout", default_value=0.0)
hp_problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=32)

hp_problem.add_hyperparameter(
    (1e-5, 1e-2, "log-uniform"), "learning_rate", default_value=0.001
)
hp_problem.add_hyperparameter(
    (1e-5, 1e-2, "log-uniform"), "base_lr", default_value=0.001
)
hp_problem.add_hyperparameter([True, False], "residual", default_value=False)

hp_problem.add_hyperparameter([True, False], "early_stopping", default_value=False)
hp_problem.add_hyperparameter((5, 20), "early_stopping_patience", default_value=5)

hp_problem.add_hyperparameter([True, False], "reduce_lr", default_value=False)
hp_problem.add_hyperparameter((0.1, 1.0), "reduce_lr_factor", default_value=0.5)
hp_problem.add_hyperparameter((5, 20), "reduce_lr_patience", default_value=5)

hp_problem.add_hyperparameter([True, False], "warmup_lr", default_value=False)
hp_problem.add_hyperparameter([True, False], "batch_normalization", default_value=False)

hp_problem.add_hyperparameter(
    ["mse", "mae", "logcosh", "mape", "msle", "huber"], "loss", default_value="mse"
)

hp_problem.add_hyperparameter(
    ["std", "minmax", "maxabs"], "scaling", default_value="std"
)


@profile
def run(config):
    return np.random.random()
