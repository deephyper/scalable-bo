import logging
import os
import pickle
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

#!!! DeepHyper Problem [START]
from deephyper.evaluator import profile
from deephyper.problem import HpProblem

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
num_layers = 7  # does not count the output layer
default_units = [1000, 1000, 1000, 500, 250, 125, 60, 30]
default_activation = ["relu", "relu", "softmax", "relu", "relu", "relu", "relu", "relu"]
for i in range(num_layers):

    if i != 1 and i != 2:  # skip attention layer
        hp_problem.add_hyperparameter(
            (10, 1024, "log-uniform"),
            f"dense_{i}:units",
            default_value=default_units[i],
        )

    if i != 2:
        hp_problem.add_hyperparameter(
            ACTIVATIONS, f"dense_{i}:activation", default_value=default_activation[i]
        )

hp_problem.add_hyperparameter(
    (10, 1024, "log-uniform"), "attention:units", default_value=1000
)
hp_problem.add_hyperparameter(
    ["softmax"], "attention:activation", default_value="softmax"
)

# Optimization hyperparameters
hp_problem.add_hyperparameter((0.1, 0.9), "momentum", default_value=0.9)
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
hp_problem.add_hyperparameter(
    (1e-5, 1e-2, "log-uniform"), "learning_rate", default_value=0.00001
)
hp_problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=32)

#!!! DeepHyper Problem [END]


@profile
def run(config):
    
    
    cols = "attention:activation,attention:units,batch_size,dense_0:activation,dense_0:units,dense_1:activation,dense_3:activation,dense_3:units,dense_4:activation,dense_4:units,dense_5:activation,dense_5:units,dense_6:activation,dense_6:units,learning_rate,momentum,optimizer"
    cols = cols.split(",") + ["objective", "elapsed"]
    config["objective"] = 0
    config["elapsed"] = 0

    df = pd.DataFrame(data=[config], columns=cols)

    model_path = os.path.join(HERE, "candle_attn_sim_surrogate.pkl")
    with open(model_path, "rb") as f:
        saved_pipeline = pickle.load(f)

    data_pipeline_model = saved_pipeline["data"]
    regr = saved_pipeline["model"]

    preprocessed_data = data_pipeline_model.transform(df)
    preds = regr.predict(preprocessed_data[:, 2:])

    preds = data_pipeline_model.transformers_[0][1].inverse_transform(preds)

    objective = preds[0,0]
    elapsed = preds[0,1]
    time.sleep(elapsed)

    return objective



if __name__ == "__main__":

    # default_config = {}
    default_config = hp_problem.default_configuration
    logging.info(f"{default_config=}")
    run(default_config)
