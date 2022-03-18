import numpy as np
from deephyper.problem import HpProblem
from deephyper.evaluator import profile

import os
import logging

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
gpu_local_idx = rank % size

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[gpu_local_idx], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_local_idx], True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

import plasma.global_vars as g

g.init_MPI()
g.conf_file = None

from plasma.conf_parser import parameters
from plasma.models.mpi_runner import mpi_train, mpi_make_predictions_and_evaluate

from plasma.preprocessor.preprocess import guarantee_preprocessed
from plasma.models.loader import Loader

import numpy as np
from deephyper.problem import HpProblem
from deephyper.evaluator import profile


hp_problem = HpProblem()
# hp_problem.add_hyperparameter((32, 256, "log-uniform"), "pred_length")
# hp_problem.add_hyperparameter((32, 256, "log-uniform"), "pred_batch_size")
# hp_problem.add_hyperparameter((32, 256, "log-uniform"), "length")
# hp_problem.add_hyperparameter((32, 256, "log-uniform"), "rnn_size")
# hp_problem.add_hyperparameter((1, 4), "rnn_layers")
# hp_problem.add_hyperparameter((32, 256, "log-uniform"), "num_conv_filters")
# hp_problem.add_hyperparameter((1, 4), "num_conv_layers")
# hp_problem.add_hyperparameter((32, 256, "log-uniform"), "dense_size")
# hp_problem.add_hyperparameter((0.0, 1.0), "regularization")
# hp_problem.add_hyperparameter((0.0, 1.0), "dense_regularization")
# hp_problem.add_hyperparameter((5e-7, 1e-4), "lr")
# hp_problem.add_hyperparameter((0.9, 1.0), "lr_decay")
# hp_problem.add_hyperparameter((0.0, 1.0), "dropout_prob")
# hp_problem.add_hyperparameter((32, 256, "log-uniform"), "batch_size")

#! test
hp_problem.add_hyperparameter((200, 201, "log-uniform"), "pred_length")
hp_problem.add_hyperparameter((128, 129, "log-uniform"), "pred_batch_size")
hp_problem.add_hyperparameter((200, 201, "log-uniform"), "length")
hp_problem.add_hyperparameter((200, 201, "log-uniform"), "rnn_size")
hp_problem.add_hyperparameter((2, 3), "rnn_layers")
hp_problem.add_hyperparameter((128, 129, "log-uniform"), "num_conv_filters")
hp_problem.add_hyperparameter((3, 4), "num_conv_layers")
hp_problem.add_hyperparameter((128, 129, "log-uniform"), "dense_size")
hp_problem.add_hyperparameter((0.001, 0.0011), "regularization")
hp_problem.add_hyperparameter((0.001, 0.0011), "dense_regularization")
hp_problem.add_hyperparameter((5e-5, 6e-5), "lr")
hp_problem.add_hyperparameter((0.97, 0.971), "lr_decay")
hp_problem.add_hyperparameter((0.1, 0.11), "dropout_prob")
hp_problem.add_hyperparameter((256, 257, "log-uniform"), "batch_size")


@profile
def run(config):
    conf = parameters(config)
    logging.info(f"conf: {conf}")

    if conf["data"]["normalizer"] == "minmax":
        from plasma.preprocessor.normalize import MinMaxNormalizer as Normalizer
    elif conf["data"]["normalizer"] == "meanvar":
        from plasma.preprocessor.normalize import MeanVarNormalizer as Normalizer
    elif conf["data"]["normalizer"] == "var":
        # performs !much better than minmaxnormalizer
        from plasma.preprocessor.normalize import VarNormalizer as Normalizer
    elif conf["data"]["normalizer"] == "averagevar":
        # performs !much better than minmaxnormalizer
        from plasma.preprocessor.normalize import AveragingVarNormalizer as Normalizer
    else:
        logging.error("wrong normalizer returning 0")
        return 0

    # set PRNG seed, unique for each worker, based on MPI task index for
    # reproducible shuffling in guranteed_preprocessed() and training steps
    custom_path = None

    #####################################################
    #                 NORMALIZATION                     #
    #####################################################
    normalizer = Normalizer(conf)

    # make sure preprocessing has been run, and results are saved to files
    # if not, only master MPI rank spawns thread pool to perform preprocessing
    (shot_list_train, shot_list_valid, shot_list_test) = guarantee_preprocessed(conf)
    # similarly, train normalizer (if necessary) w/ master MPI rank only
    normalizer.train()  # verbose=False only suppresses if purely loading

    logging.info("begin preprocessor+normalization (all MPI ranks)...")
    # second call has ALL MPI ranks load preprocessed shots from .npz files
    (shot_list_train, shot_list_valid, shot_list_test) = guarantee_preprocessed(
        conf, verbose=False
    )
    # second call to normalizer training
    normalizer.conf["data"]["recompute_normalization"] = False
    normalizer.train(verbose=False)
    # KGF: may want to set it back...
    # normalizer.conf['data']['recompute_normalization'] = conf['data']['recompute_normalization']   # noqa
    loader = Loader(conf, normalizer)
    logging.info("preprocessor+normalization...done")

    # TODO(KGF): both preprocess.py and normalize.py are littered with print()
    # calls that should probably be replaced with print_unique() when they are not
    # purely loading previously-computed quantities from file
    # (or we can continue to ensure that they are only ever executed by 1 rank)

    #####################################################
    #                    TRAINING                       #
    #####################################################

    mpi_train(
        conf,
        shot_list_train,
        shot_list_valid,
        loader,
        shot_list_test=shot_list_test,
    )

    # load last model for testing
    loader.set_inference_mode(True)

    (
        y_prime_valid,
        y_gold_valid,
        disruptive_valid,
        roc_valid,
        loss_valid,
    ) = mpi_make_predictions_and_evaluate(conf, shot_list_valid, loader, custom_path)

    logging.info("finished.")
    return roc_valid
