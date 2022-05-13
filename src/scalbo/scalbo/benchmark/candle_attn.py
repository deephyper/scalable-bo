from __future__ import print_function

import numpy as np
import sklearn
import h5py
import pathlib

import argparse
import os
import logging
import warnings
import json


if __name__ == "__main__":
    rank = 0
    gpu_local_idx = 0
else:  # Assuming a ThetaGPU Node here
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    gpu_local_idx = rank % 8

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[gpu_local_idx], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_local_idx], True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        logging.info(
            f"[r={rank}]: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU"
        )
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        logging.info(f"{e}")

import tensorflow.keras as ke
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import (
    Callback,
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.metrics import (
    auc,
    roc_curve,
    f1_score,
    precision_recall_curve,
    accuracy_score,
)

import attn
import candle

import attn_viz_utils as attnviz

#!!! DeepHyper Problem [START]
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

np.set_printoptions(precision=4)
tf.compat.v1.disable_eager_execution()


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def tf_auc(y_true, y_pred):
    auc = tf.compat.v1.metrics.auc(y_true, y_pred)[1]
    tf.compat.v1.keras.backend.get_session().run(
        tf.compat.v1.local_variables_initializer()
    )
    return auc


def auroc(y_true, y_pred):
    score = tf.py_func(
        lambda y_true, y_pred: roc_auc_score(
            y_true, y_pred, average="macro", sample_weight=None
        ).astype("float32"),
        [y_true, y_pred],
        "float32",
        stateful=False,
        name="sklearnAUC",
    )
    return score


def covariance(x, y):
    return K.mean(x * y) - K.mean(x) * K.mean(y)


def corr(y_true, y_pred):
    cov = covariance(y_true, y_pred)
    var1 = covariance(y_true, y_true)
    var2 = covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


class LoggingCallback(Callback):
    def __init__(self, print_fcn=logging.info):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (
            epoch,
            ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())),
        )
        self.print_fcn(msg)


def build_type_classifier(x_train, y_train, x_test, y_test):
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    from xgboost import XGBClassifier

    clf = XGBClassifier(max_depth=6, n_estimators=100)
    clf.fit(
        x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=False
    )
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(acc)
    return clf


def initialize_parameters(default_model="attn_default_model.txt"):

    # Build benchmark object
    attnBmk = attn.BenchmarkAttn(
        attn.file_path,
        default_model,
        "keras",
        prog="attn_baseline",
        desc="Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(attnBmk)

    return gParameters


def save_cache(
    cache_file, x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels
):
    with h5py.File(cache_file, "w") as hf:
        hf.create_dataset("x_train", data=x_train)
        hf.create_dataset("y_train", data=y_train)
        hf.create_dataset("x_val", data=x_val)
        hf.create_dataset("y_val", data=y_val)
        hf.create_dataset("x_test", data=x_test)
        hf.create_dataset("y_test", data=y_test)
        hf.create_dataset(
            "x_labels",
            (len(x_labels), 1),
            "S100",
            data=[x.encode("ascii", "ignore") for x in x_labels],
        )
        hf.create_dataset(
            "y_labels",
            (len(y_labels), 1),
            "S100",
            data=[x.encode("ascii", "ignore") for x in y_labels],
        )


def load_cache(cache_file):
    with h5py.File(cache_file, "r") as hf:
        x_train = hf["x_train"][:]
        y_train = hf["y_train"][:]
        x_val = hf["x_val"][:]
        y_val = hf["y_val"][:]
        x_test = hf["x_test"][:]
        y_test = hf["y_test"][:]
        x_labels = [x[0].decode("unicode_escape") for x in hf["x_labels"][:]]
        y_labels = [x[0].decode("unicode_escape") for x in hf["y_labels"][:]]
    return x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels


def build_attention_model(params, PS):

    assert len(params["dense"]) == len(params["activation"])
    assert len(params["dense"]) > 3

    DR = params["dropout"]
    inputs = Input(shape=(PS,))
    x = Dense(params["dense"][0], activation=params["activation"][0])(inputs)
    x = BatchNormalization()(x)
    a = Dense(params["dense"][1], activation=params["activation"][1])(x)
    a = BatchNormalization()(a)
    b = Dense(params["dense"][2], activation=params["activation"][2])(x)
    x = ke.layers.multiply([a, b])

    for i in range(3, len(params["dense"]) - 1):
        x = Dense(params["dense"][i], activation=params["activation"][i])(x)
        x = BatchNormalization()(x)
        x = Dropout(DR)(x)

    outputs = Dense(params["dense"][-1], activation=params["activation"][-1])(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def run_candle(params):
    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)

    # cache data
    if os.path.exists("/dev/shm"):
        train_file = params["train_data"]
        cache_train_file = os.path.join("/dev/shm", os.path.basename(train_file))
        # only the first rank of each node caches the data
        if os.path.exists(cache_train_file):
            params["train_data"] = cache_train_file
        else:
            if rank % 16 == 0 and os.path.exists(train_file):
                ret = os.system(f"cp {train_file} {cache_train_file}")
                if ret == 0:
                    params["train_data"] = cache_train_file
                else:
                    params["train_data"] = train_file

    # Construct extension to save model
    ext = attn.extension_from_parameters(params, "keras")
    candle.verify_path(params["save_path"])
    prefix = "{}{}".format(params["save_path"], ext)
    logfile = params["logfile"] if params["logfile"] else prefix + ".log"
    root_fname = params.get("root_fname", "Agg_attn_bin")
    candle.set_up_logger(logfile, attn.logger, params["verbose"])
    attn.logger.info("Params: {}".format(params))

    # Get default parameters for initialization and optimizer functions
    # keras_defaults = candle.keras_default_config()

    ##
    X_train, _Y_train, X_val, _Y_val, X_test, _Y_test = attn.load_data(params, seed)

    # move this inside the load_data function
    Y_train = _Y_train["AUC"]
    Y_test = _Y_test["AUC"]
    Y_val = _Y_val["AUC"]

    Y_train_neg, Y_train_pos = np.bincount(Y_train)
    Y_test_neg, Y_test_pos = np.bincount(Y_test)
    Y_val_neg, Y_val_pos = np.bincount(Y_val)

    Y_train_total = Y_train_neg + Y_train_pos
    Y_test_total = Y_test_neg + Y_test_pos
    Y_val_total = Y_val_neg + Y_val_pos

    total = Y_train_total + Y_test_total + Y_val_total
    # neg = Y_train_neg + Y_test_neg + Y_val_neg
    pos = Y_train_pos + Y_test_pos + Y_val_pos

    logging.info(
        "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
            total, pos, 100 * pos / total
        )
    )

    nb_classes = params["dense"][-1]

    Y_train = to_categorical(Y_train, nb_classes)
    Y_test = to_categorical(Y_test, nb_classes)
    Y_val = to_categorical(Y_val, nb_classes)

    y_integers = np.argmax(Y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_integers), y=y_integers
    )
    d_class_weights = dict(enumerate(class_weights))

    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}")

    logging.info(f"Y_train shape: {Y_train.shape}")
    logging.info(f"Y_test shape: {Y_test.shape}")

    PS = X_train.shape[1]
    model = build_attention_model(params, PS)

    # TODO: load checkpointed weights
    if "cp_weights_path" in params:
        model.load_weights(params["cp_weights_path"])

    kerasDefaults = candle.keras_default_config()
    if params["momentum"]:
        kerasDefaults["momentum_sgd"] = params["momentum"]

    optimizer = candle.build_optimizer(
        params["optimizer"], params["learning_rate"], kerasDefaults
    )

    model.compile(loss=params["loss"], optimizer=optimizer, metrics=["acc", tf_auc])

    # set up a bunch of callbacks to do work during model training..

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(params["save_path"], root_fname + ".autosave.model.h5"),
        verbose=params.get("verbose", 0),
        save_weights_only=False,
        save_best_only=True,
    )
    csv_logger = CSVLogger("{}/{}.training.log".format(params["save_path"], root_fname))
    reduce_lr = ReduceLROnPlateau(
        monitor="val_tf_auc",
        factor=0.20,
        patience=40,
        verbose=params.get("verbose", 0),
        mode="auto",
        min_delta=0.0001,
        cooldown=3,
        min_lr=0.000000001,
    )
    early_stop = EarlyStopping(
        monitor="val_tf_auc",
        patience=200,
        verbose=params.get("verbose", 0),
        mode="auto",
    )
    candle_monitor = candle.CandleRemoteMonitor(params=params)

    candle_monitor = candle.CandleRemoteMonitor(params=params)
    timeout_monitor = candle.TerminateOnTimeOut(params["timeout"])
    tensorboard = TensorBoard(log_dir="tb/tb{}".format(ext))

    history_logger = LoggingCallback(attn.logger.debug)

    # callbacks = [candle_monitor, timeout_monitor, history_logger]
    callbacks = [timeout_monitor, history_logger]

    if params["reduce_lr"]:
        callbacks.append(reduce_lr)

    if params.get("csv_logger", False):
        callbacks.append(csv_logger)

    if params["use_cp"]:
        callbacks.append(checkpointer)
    if params["use_tb"]:
        callbacks.append(tensorboard)
    if params["early_stop"]:
        callbacks.append(early_stop)

    epochs = params["epochs"]
    batch_size = params["batch_size"]
    history = model.fit(
        X_train,
        Y_train,
        class_weight=d_class_weights,
        batch_size=batch_size,
        epochs=epochs,
        verbose=params.get("verbose", 0),
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
    )

    # diagnostic plots
    if params.get("evaluate_model", False):
        if "loss" in history.history.keys():
            candle.plot_history(params["save_path"] + root_fname, history, "loss")
        if "acc" in history.history.keys():
            candle.plot_history(params["save_path"] + root_fname, history, "acc")
        if "tf_auc" in history.history.keys():
            candle.plot_history(params["save_path"] + root_fname, history, "tf_auc")

        # Evaluate model
        score = model.evaluate(X_test, Y_test, verbose=0)
        Y_predict = model.predict(X_test)

        evaluate_model(
            params,
            root_fname,
            nb_classes,
            Y_test,
            _Y_test,
            Y_predict,
            pos,
            total,
            score,
        )

        save_and_test_saved_model(params, model, root_fname, X_train, X_test, Y_test)

        attn.logger.handlers = []

    return history


def evaluate_model(
    params, root_fname, nb_classes, Y_test, _Y_test, Y_predict, pos, total, score
):

    threshold = 0.5

    Y_pred_int = (Y_predict[:, 0] < threshold).astype(np.int)
    Y_test_int = (Y_test[:, 0] < threshold).astype(np.int)

    logging.info("creating table of predictions")
    f = open(params["save_path"] + root_fname + ".predictions.tsv", "w")
    for index, row in _Y_test.iterrows():
        if row["AUC"] == 1:
            if Y_pred_int[index] == 1:
                call = "TP"
            else:
                call = "FN"
        if row["AUC"] == 0:
            if Y_pred_int[index] == 0:
                call = "TN"
            else:
                call = "FP"
        # 1 TN 0 0.6323 NCI60.786-0 NSC.256439 NSC.102816
        logging.info(
            index,
            "\t",
            call,
            "\t",
            Y_pred_int[index],
            "\t",
            row["AUC"],
            "\t",
            row["Sample"],
            "\t",
            row["Drug1"],
            file=f,
        )
    f.close()

    false_pos_rate, true_pos_rate, thresholds = roc_curve(Y_test[:, 0], Y_predict[:, 0])
    roc_auc = auc(false_pos_rate, true_pos_rate)

    auc_keras = roc_auc
    fpr_keras = false_pos_rate
    tpr_keras = true_pos_rate

    # ROC plots
    fname = params["save_path"] + root_fname + ".auroc.pdf"
    logging.info(f"creating figure at {fname}")
    attnviz.plot_ROC(fpr_keras, tpr_keras, auc_keras, fname)
    # Zoom in view of the upper left corner.
    fname = params["save_path"] + root_fname + ".auroc_zoom.pdf"
    logging.info(f"creating figure at {fname}")
    attnviz.plot_ROC(fpr_keras, tpr_keras, auc_keras, fname, zoom=True)

    f1 = f1_score(Y_test_int, Y_pred_int)

    precision, recall, thresholds = precision_recall_curve(
        Y_test[:, 0], Y_predict[:, 0]
    )
    pr_auc = auc(recall, precision)

    pr_keras = pr_auc
    precision_keras = precision
    recall_keras = recall

    logging.info("f1=%.3f auroc=%.3f aucpr=%.3f" % (f1, auc_keras, pr_keras))
    # Plot RF
    fname = params["save_path"] + root_fname + ".aurpr.pdf"
    logging.info(f"creating figure at {fname}")
    no_skill = len(Y_test_int[Y_test_int == 1]) / len(Y_test_int)
    attnviz.plot_RF(recall_keras, precision_keras, pr_keras, no_skill, fname)

    # Compute confusion matrix
    cnf_matrix = sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int)
    # Plot non-normalized confusion matrix
    class_names = ["Non-Response", "Response"]
    fname = params["save_path"] + root_fname + ".confusion_without_norm.pdf"
    logging.info(f"creating figure at {fname}")
    attnviz.plot_confusion_matrix(
        cnf_matrix,
        fname,
        classes=class_names,
        title="Confusion matrix, without normalization",
    )
    # Plot normalized confusion matrix
    fname = params["save_path"] + root_fname + ".confusion_with_norm.pdf"
    logging.info(f"creating figure at {fname}")
    attnviz.plot_confusion_matrix(
        cnf_matrix,
        fname,
        classes=class_names,
        normalize=True,
        title="Normalized confusion matrix",
    )

    logging.info(
        "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
            total, pos, 100 * pos / total
        )
    )

    logging.info(sklearn.metrics.roc_auc_score(Y_test_int, Y_pred_int))
    logging.info(sklearn.metrics.balanced_accuracy_score(Y_test_int, Y_pred_int))
    logging.info(sklearn.metrics.classification_report(Y_test_int, Y_pred_int))
    logging.info(sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int))
    logging.info("score")
    logging.info(score)

    logging.info(f"Test val_loss: {score[0]}")
    logging.info(f"Test accuracy: {score[1]}")


def save_and_test_saved_model(params, model, root_fname, X_train, X_test, Y_test):

    # serialize model to JSON
    model_json = model.to_json()
    with open(params["save_path"] + root_fname + ".model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(params["save_path"] + root_fname + ".model.h5")
    logging.info("Saved model to disk")

    # load json and create model
    json_file = open(params["save_path"] + root_fname + ".model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model_json.load_weights(params["save_path"] + root_fname + ".model.h5")
    logging.info("Loaded json model from disk")

    # evaluate json loaded model on test data
    loaded_model_json.compile(
        loss="binary_crossentropy", optimizer=params["optimizer"], metrics=["accuracy"]
    )
    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

    logging.info("json Validation loss:", score_json[0])
    logging.info("json Validation accuracy:", score_json[1])

    logging.info(
        "json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1] * 100)
    )

    # predict using loaded model on test and training data
    predict_train = loaded_model_json.predict(X_train)
    predict_test = loaded_model_json.predict(X_test)
    logging.info("train_shape:", predict_train.shape)
    logging.info("test_shape:", predict_test.shape)

    predict_train_classes = np.argmax(predict_train, axis=1)
    predict_test_classes = np.argmax(predict_test, axis=1)
    np.savetxt(
        params["save_path"] + root_fname + "_predict_train.csv",
        predict_train,
        delimiter=",",
        fmt="%.3f",
    )
    np.savetxt(
        params["save_path"] + root_fname + "_predict_test.csv",
        predict_test,
        delimiter=",",
        fmt="%.3f",
    )

    np.savetxt(
        params["save_path"] + root_fname + "_predict_train_classes.csv",
        predict_train_classes,
        delimiter=",",
        fmt="%d",
    )
    np.savetxt(
        params["save_path"] + root_fname + "_predict_test_classes.csv",
        predict_test_classes,
        delimiter=",",
        fmt="%d",
    )


@profile
def run(config, log_dir=None, cache_data=False):
    params = initialize_parameters()

    params["epochs"] = 10
    params["timeout"] = 60 * 10  # 10 minutes per model

    if log_dir is not None:
        params["save_path"] = os.path.join(log_dir, "save")
        pathlib.Path(params["save_path"]).mkdir(parents=True, exist_ok=True)

        if "trial_id" in config:
            params["root_fname"] = f"trial-{config['trial_id']}"
            params["use_cp"] = True  # checkpointing
            params["epochs"] = 1

            # params["save_path"] + root_fname + ".autosave.model.h5"
            if config["resource"] > 1:
                params["cp_weights_path"] = os.path.join(
                    params["save_path"], params["root_fname"] + ".autosave.model.h5"
                )
        else:
            params["use_cp"] = False
            params["root_fname"] = f"job-{config['job_id']}"

    if len(config) > 0:
        # collect dense units
        dense_keys = ["dense_0:units", "attention:units", "attention:units"] + [
            f"dense_{i}:units" for i in range(3, num_layers)
        ]
        dense = [config[k] for k in dense_keys] + [2]
        config = {k: v for k, v in config.items() if not (k in dense_keys)}
        params["dense"] = dense

        # collect activation units
        activation_keys = [
            "dense_0:activation",
            "dense_1:activation",
            "attention:activation",
        ] + [f"dense_{i}:activation" for i in range(3, num_layers)]
        activation = [config[k] for k in activation_keys] + ["softmax"]
        config = {k: v for k, v in config.items() if not (k in activation_keys)}
        params["activation"] = activation

        params.update(config)

    try:
        history = run_candle(params)
        score = history.history["val_tf_auc"][-1]
    except Exception as e:
        score = 0

    return score


def full_training(config):

    config["epochs"] = 100
    config["timeout"] = 60 * 60 * 1  # 1 hour
    config["evaluate_model"] = True

    run(config, cache_data=True)

def create_parser():
    parser = argparse.ArgumentParser(description="ECP-Candle Attn Benchmark Parser.")

    parser.add_argument(
        "--json", 
        type=str, 
        default=None, 
        help="Path to the JSON file containing configuration to test."
    )
    return parser



def load_json(f):
    with open(f, "r") as f:
        js_data = json.load(f)
    return js_data


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    if args.json:
        filtered_keys = ["trial_id", "resource"]
        config = load_json(args.json)["0"]
        config = {k:v for k,v in config.items() if not(k in filtered_keys)}
    else:
        # default_config = {}
        config = hp_problem.default_configuration

    full_training(config)
