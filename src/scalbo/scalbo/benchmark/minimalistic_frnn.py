import os
import logging
import pathlib
import numpy as np
import time
import pprint
import itertools
from functools import partial
import json

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
gpu_per_node = 8
gpu_local_idx = rank % gpu_per_node
node = int(rank / gpu_per_node)

# dataset_path = "/lus/theta-fs0/projects/fusiondl_aesp/felker"
# sub_dirs = [
#     "normalization",
#     "processed_shotlists",
#     "processed_shots",
#     "shot_lists",
# ]
# sub_files = [
#     "normalization_signal_group_274046652389426782036862662489435313687-old.npz",
# ]
# if gpu_local_idx == 0:
#     pathlib.Path(f"/dev/shm/{node}").mkdir(parents=True, exist_ok=True)
#     for sub_dir in sub_dirs:
#         os.system(f"cp -r {dataset_path}/{sub_dir} /dev/shm/{node}/{sub_dir}")
#     for sub_file in sub_files:
#         os.system(f"cp {dataset_path}/{sub_file} /dev/shm/{node}/{sub_file}")

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.backend import clear_session


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[gpu_local_idx], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_local_idx], True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        logging.info(f"[r={rank}]: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        logging.info(f"{e}")

from tensorflow.keras.layers import (
    Input,
    Dense, Activation, Dropout, Lambda,
    Reshape, Flatten, Permute,  # RepeatVector
    LSTM, SimpleRNN, BatchNormalization,
    Convolution1D, MaxPooling1D, TimeDistributed,
    Concatenate
    )
CuDNNLSTM = LSTM
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.optimizers import (SGD, Adam, RMSprop, Nadam)
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2  # l1, l1_l2
from tensorflow.keras.metrics import AUC

# from plasma.models.tcn import TCN
from plasma.conf_parser import parameters
from plasma.preprocessor.preprocess import guarantee_preprocessed
from plasma.models.loader import Loader
from plasma.utils.processing import concatenate_sublists
from plasma.utils.performance import PerformanceAnalyzer
from plasma.utils.evaluation import get_loss_from_list

import plasma.global_vars as g
g.init_MPI()


from deephyper.problem import HpProblem
from deephyper.evaluator import profile


hp_problem = HpProblem()
hp_problem.add_hyperparameter((32, 256, "log-uniform"), "pred_batch_size")
hp_problem.add_hyperparameter((32, 256, "log-uniform"), "length")
hp_problem.add_hyperparameter((32, 256, "log-uniform"), "rnn_size")
hp_problem.add_hyperparameter((1, 4), "rnn_layers")
hp_problem.add_hyperparameter((32, 256, "log-uniform"), "num_conv_filters")
hp_problem.add_hyperparameter((1, 4), "num_conv_layers")
hp_problem.add_hyperparameter((32, 256, "log-uniform"), "dense_size")
hp_problem.add_hyperparameter((0.0, 1.0), "regularization")
hp_problem.add_hyperparameter((0.0, 1.0), "dense_regularization")
hp_problem.add_hyperparameter((1e-7, 1e-3, "log-uniform"), "lr")
hp_problem.add_hyperparameter((0.9, 1.0), "lr_decay")
hp_problem.add_hyperparameter((0.9, 1.0), "momentum")
hp_problem.add_hyperparameter((0.0, 0.5), "dropout_prob")
hp_problem.add_hyperparameter((32, 512, "log-uniform"), "batch_size")


class ResetStatesCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        self.model.reset_states()


class LayerReset(tf.keras.layers.Layer):
    def __init__(self, layer):
        super(LayerReset, self).__init__()
        self._layer = layer

    def call(self, batches_to_reset):
        def reset_states(layer, batches_to_reset):
            for j, reset in enumerate(tf.unstack(batches_to_reset)):
                if reset != 0:
                    for state in layer.states:
                        assert len(batches_to_reset) == state.shape[0]
                        batch_states = tf.keras.backend.get_value(state)
                        batch_states[j] = 0
                        tf.keras.backend.set_value(state, batch_states)
        # reset_states(self._layer, batches_to_reset)
        return batches_to_reset


class ModelBuilder(object):
    def __init__(self, conf):
        self.conf = conf

    def get_0D_1D_indices(self):
        # make sure all 1D indices are contiguous in the end!
        use_signals = self.conf['paths']['use_signals']
        indices_0d = []
        indices_1d = []
        num_0D = 0
        num_1D = 0
        curr_idx = 0
        is_1D_region = use_signals[0].num_channels > 1
        for sig in use_signals:
            num_channels = sig.num_channels
            indices = range(curr_idx, curr_idx+num_channels)
            if num_channels > 1:
                indices_1d += indices
                num_1D += 1
                is_1D_region = True
            else:
                assert not is_1D_region, ("Check that use_signals are ordered with 1D signals last!")
                assert num_channels == 1
                indices_0d += indices
                num_0D += 1
                is_1D_region = False
            curr_idx += num_channels
        return (np.array(indices_0d).astype(np.int32), np.array(indices_1d).astype(np.int32), num_0D, num_1D)

    def build_model(self):
        conf = self.conf
        model_conf = conf['model']
        rnn_size = model_conf['rnn_size']
        rnn_type = model_conf['rnn_type']
        regularization = model_conf['regularization']
        dense_regularization = model_conf['dense_regularization']
        use_batch_norm = model_conf.get('use_batch_norm', False)

        dropout_prob = model_conf['dropout_prob']
        length = model_conf['length']
        stateful = model_conf['stateful']
        return_sequences = model_conf['return_sequences']
        output_activation = conf['data']['target'].activation
        use_signals = conf['paths']['use_signals']
        num_signals = sum([sig.num_channels for sig in use_signals])
        num_conv_filters = model_conf['num_conv_filters']
        num_conv_layers = model_conf['num_conv_layers']
        size_conv_filters = model_conf['size_conv_filters']
        pool_size = model_conf['pool_size']
        dense_size = model_conf['dense_size']

        batch_size = conf['training']['batch_size']

        if rnn_type == 'LSTM':
            rnn_model = LSTM
        elif rnn_type == 'SimpleRNN':
            rnn_model = SimpleRNN
        else:
            rnn_model = CuDNNLSTM

        batch_input_shape = (batch_size, length, num_signals)

        indices_0d, indices_1d, num_0D, num_1D = self.get_0D_1D_indices()

        # ==========
        # PRE_RNN
        # ==========
        pre_rnn_input = Input(shape=(num_signals,))

        if num_1D > 0:
            pre_rnn_1D = Lambda(lambda x: x[:, len(indices_0d):], output_shape=(len(indices_1d),))(pre_rnn_input)
            pre_rnn_0D = Lambda(lambda x: x[:, :len(indices_0d)], output_shape=(len(indices_0d),))(pre_rnn_input)
            pre_rnn_1D = Reshape((num_1D, len(indices_1d)//num_1D))(pre_rnn_1D)
            pre_rnn_1D = Permute((2, 1))(pre_rnn_1D)
            if model_conf.get('simple_conv', False):
                for i in range(num_conv_layers):
                    pre_rnn_1D = Convolution1D(num_conv_filters, size_conv_filters, padding='valid', activation='relu')(pre_rnn_1D)
                pre_rnn_1D = MaxPooling1D(pool_size)(pre_rnn_1D)
            else:
                for i in range(num_conv_layers):
                    div_fac = 2**i
                    pre_rnn_1D = Convolution1D(num_conv_filters//div_fac, size_conv_filters, padding='valid')(pre_rnn_1D)
                    if use_batch_norm:
                        pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
                        pre_rnn_1D = Activation('relu')(pre_rnn_1D)
                    pre_rnn_1D = Convolution1D(num_conv_filters//div_fac, 1, padding='valid')(pre_rnn_1D)
                    if use_batch_norm:
                        pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
                        pre_rnn_1D = Activation('relu')(pre_rnn_1D)
                    pre_rnn_1D = MaxPooling1D(pool_size)(pre_rnn_1D)
            pre_rnn_1D = Flatten()(pre_rnn_1D)
            pre_rnn_1D = Dense(dense_size, kernel_regularizer=l2(dense_regularization), bias_regularizer=l2(dense_regularization), activity_regularizer=l2(dense_regularization))(pre_rnn_1D)
            if use_batch_norm:
                pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
            pre_rnn_1D = Activation('relu')(pre_rnn_1D)
            pre_rnn_1D = Dense(dense_size//4, kernel_regularizer=l2(dense_regularization), bias_regularizer=l2(dense_regularization), activity_regularizer=l2(dense_regularization))(pre_rnn_1D)
            if use_batch_norm:
                pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
            pre_rnn_1D = Activation('relu')(pre_rnn_1D)
            pre_rnn = Concatenate()([pre_rnn_0D, pre_rnn_1D])
        else:
            pre_rnn = pre_rnn_input

        if model_conf['rnn_layers'] == 0 or model_conf.get('extra_dense_input', False):
            pre_rnn = Dense(dense_size, activation='relu', kernel_regularizer=l2(dense_regularization), bias_regularizer=l2(dense_regularization), activity_regularizer=l2(dense_regularization))(pre_rnn)
            pre_rnn = Dense(dense_size//2, activation='relu', kernel_regularizer=l2(dense_regularization), bias_regularizer=l2(dense_regularization), activity_regularizer=l2(dense_regularization))(pre_rnn)
            pre_rnn = Dense(dense_size//4, activation='relu', kernel_regularizer=l2(dense_regularization), bias_regularizer=l2(dense_regularization), activity_regularizer=l2(dense_regularization))(pre_rnn)

        pre_rnn_model = tf.keras.Model(inputs=pre_rnn_input, outputs=pre_rnn)
        pre_rnn_model.summary()
        

        x_input = Input(batch_shape=batch_input_shape)
        if num_1D > 0 or model_conf.get('extra_dense_input', False):
            x_in = TimeDistributed(pre_rnn_model)(x_input)
        else:
            x_in = x_input
        
        # ==========
        # RNN MODEL
        # ==========
        model_kwargs = dict(return_sequences=return_sequences,
                            # batch_input_shape=batch_input_shape,
                            stateful=stateful,
                            kernel_regularizer=l2(regularization),
                            recurrent_regularizer=l2(regularization),
                            bias_regularizer=l2(regularization),
                            )
        if rnn_type != 'CuDNNLSTM':
            # recurrent_dropout is unsupported in cuDNN library
            model_kwargs['recurrent_dropout'] = dropout_prob  # recurrent states
        model_kwargs['dropout'] = dropout_prob  # input states
        for _ in range(model_conf['rnn_layers']):
            rnn_layer = rnn_model(rnn_size, **model_kwargs)
            x_in = rnn_layer(x_in)
            x_in = Dropout(dropout_prob)(x_in)
        if return_sequences:
            x_out = TimeDistributed(Dense(1, activation=output_activation))(x_in)

        model = tf.keras.Model(inputs=x_input, outputs=x_out)
        # bug with tensorflow/Keras
        # TODO(KGF): what is this bug? this is the only direct "tensorflow"
        # import outside of mpi_runner.py and runner.py
        # if (conf['model']['backend'] == 'tf'
        #         or conf['model']['backend'] == 'tensorflow'):
        #     first_time = "tensorflow" not in sys.modules
        #     import tensorflow as tf
        #     if first_time:
        #         tf.compat.v1.keras.backend.get_session().run(
        #             tf.global_variables_initializer())
        model.reset_states()
        return model
    
    def build_model_zeros(self):
        conf = self.conf
        model_conf = conf['model']
        length = model_conf['length']
        use_signals = conf['paths']['use_signals']
        num_signals = sum([sig.num_channels for sig in use_signals])
        batch_size = conf['training']['batch_size']
        batch_input_shape = (batch_size, length, num_signals)
        batch_output_shape = (batch_size, length, 1)
        x_input = Input(batch_shape=batch_input_shape)

        output = output = Lambda(lambda _: tf.zeros(batch_output_shape))(x_input)

        model = tf.keras.Model(inputs=x_input, outputs=output)
        return model
    
    def build_model_ones(self):
        conf = self.conf
        model_conf = conf['model']
        length = model_conf['length']
        use_signals = conf['paths']['use_signals']
        num_signals = sum([sig.num_channels for sig in use_signals])
        batch_size = conf['training']['batch_size']
        batch_input_shape = (batch_size, length, num_signals)
        batch_output_shape = (batch_size, length, 1)
        x_input = Input(batch_shape=batch_input_shape)

        output = output = Lambda(lambda _: tf.ones(batch_output_shape))(x_input)

        model = tf.keras.Model(inputs=x_input, outputs=output)
        return model

    def build_optimizer(self):
        conf = self.conf
        lr = conf['model']["lr"]
        momentum = conf['model']["momentum"]
        clipnorm = conf['model']["clipnorm"]
        optimizers = {
            "sgd": SGD,
            "momentum_sgd": SGD,
            "adam": Adam,
            "rmsprop": RMSprop,
            "nadam": Nadam,
        }
        optimizers_kwargs = {
            "sgd": {
                "learning_rate": lr,
                "clipnorm": clipnorm,
            },
            "momentum_sgd": {
                "learning_rate": lr,
                "clipnorm": clipnorm,
                "decay": 1e-6,
                "momentum": momentum,
            },
            "adam": {
                "learning_rate": lr,
                "clipnorm": clipnorm,
            },
            "rmsprop": {
                "learning_rate": lr,
                "clipnorm": clipnorm,
            },
            "nadam": {
                "learning_rate": lr,
                "clipnorm": clipnorm,
            },
        }
        
        opt_kwargs = optimizers_kwargs.get(conf['model']['optimizer'], optimizers_kwargs['adam'])
        optimizer = optimizers.get(conf['model']['optimizer'], optimizers['adam'])(**opt_kwargs)

        return optimizer


class DataHandler(object):
    def __init__(self, conf):
        self.conf = conf
    
    def build_loader(self):
        conf = self.conf
        if conf['data']['normalizer'] == 'minmax':
            from plasma.preprocessor.normalize import MinMaxNormalizer as Normalizer
        elif conf['data']['normalizer'] == 'meanvar':
            from plasma.preprocessor.normalize import MeanVarNormalizer as Normalizer
        elif conf['data']['normalizer'] == 'averagevar':
            from plasma.preprocessor.normalize import AveragingVarNormalizer as Normalizer
        else: # conf['data']['normalizer'] == 'var':
            from plasma.preprocessor.normalize import VarNormalizer as Normalizer

        normalizer = Normalizer(conf)
        normalizer.train()
        loader = Loader(conf, normalizer)
        loader.verbose = False
        return loader

    def load_dataset(self, shot_list, loader):
        conf = self.conf
        batch_generator = partial(loader.training_batch_generator, shot_list=shot_list)

        length = conf['model']['length']
        use_signals = conf['paths']['use_signals']
        num_signals = sum([sig.num_channels for sig in use_signals])
        batch_size = conf['training']['batch_size']
        
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, length, num_signals), dtype=tf.float32),
                # (
                #     tf.TensorSpec(shape=(batch_size, length, num_signals), dtype=tf.float32),
                #     tf.TensorSpec(shape=(batch_size, 1), dtype=tf.int32),
                # ),
                tf.TensorSpec(shape=(batch_size, length, 1), dtype=tf.float32),
            )
        )
        return dataset

class ModelEvaluator(object):
    def __init__(self, model, loader, conf) -> None:
        self.model = model
        self.loader = loader
        self.conf = conf

    def make_predictions(self, shot_list):
        model = self.model
        loader = self.loader
        conf = self.conf

        loader.set_inference_mode(True)
        np.random.seed(g.task_index)
        shot_list.sort()

        y_prime = []
        y_gold = []
        disruptive = []

        model.reset_states()
        shot_sublists = shot_list.sublists(conf['model']['pred_batch_size'], do_shuffle=False, equal_size=True)
        y_prime_global = []
        y_gold_global = []
        disruptive_global = []
        if g.task_index != 0:
            loader.verbose = False

        color = 2
        for (i, shot_sublist) in enumerate(shot_sublists):
            shpz = []
            max_length = -1 # So non shot predictive workers don't have a real length
            if i % g.num_workers == g.task_index:
                color = 1
                X, y, shot_lengths, disr = loader.load_as_X_y_pred(shot_sublist)

                # load data and fit on data
                y_p = model.predict(X, batch_size=conf['model']['pred_batch_size'])
                model.reset_states()
                y_p = loader.batch_output_to_array(y_p)
                y = loader.batch_output_to_array(y)

                # cut arrays back
                y_p = [arr[:shot_lengths[j]] for (j, arr) in enumerate(y_p)]
                y = [arr[:shot_lengths[j]] for (j, arr) in enumerate(y)]

                y_prime += y_p
                y_gold += y
                disruptive += disr

            if (i % g.num_workers == g.num_workers - 1
                    or i == len(shot_sublists) - 1):
                # Create numpy block from y list which is used in MPI
                # Pads y_prime and y_gold with zeros to maximum shot length within block being transferred
                if color ==1:
                    shpz = [max(y.shape) for y in y_prime]
                    max_length = max([max(y.shape) for y in y_p])
                max_length = g.comm.allreduce(max_length, MPI.MAX)
                if color == 1:
                    y_prime_numpy = np.stack([np.pad(sublist, pad_width=((0,max_length-max(sublist.shape)),(0,0))) for sublist in y_prime])
                    y_gold_numpy = np.stack([np.pad(sublist, pad_width=((0,max_length-max(sublist.shape)),(0,0))) for sublist in y_gold])

                temp_predictor_only_comm = MPI.Comm.Split(g.comm, color, i)
                # Create numpy array to store all processors output, then aggregate and unpad using MPI gathered shape list
                shpzg = g.comm.allgather(shpz)
                shpzg = list(itertools.chain(*shpzg))
                shpzg = [s for s in shpzg if s != []]
                max_length = g.comm.allreduce(max_length, MPI.MAX)
                if color == 1:
                    num_pred = temp_predictor_only_comm.size
                else:
                    num_pred = g.comm.size - temp_predictor_only_comm.size
                y_primeg = np.zeros((num_pred*conf['model']['pred_batch_size'],max_length,1), dtype=conf['data']['floatx'])
                y_goldg  = np.zeros((num_pred*conf['model']['pred_batch_size'],max_length,1), dtype=conf['data']['floatx'])
                y_primeg_flattend = np.zeros(y_primeg.flatten().shape)
                y_goldg_flattend  = np.zeros(y_goldg.flatten().shape)
                if color == 1:
                    # Ensure that numpy arrays have correct dimensions before gathering them
                    assert num_pred*max(y_prime_numpy.flatten().shape) == max(y_primeg_flattend.shape)
                    assert num_pred*max(y_gold_numpy.flatten().shape) == max(y_goldg_flattend.shape)
                    temp_predictor_only_comm.Allgather(y_prime_numpy.flatten(), y_primeg_flattend)
                    temp_predictor_only_comm.Allgather(y_gold_numpy.flatten(), y_goldg_flattend)
                # Process 0 broadcast y_primeg and y_goldg to all processors, including ones
                # not involved in calculating predictions so they can each create their own
                # y_prime_global and y_gold_global
                g.comm.Barrier()
                g.comm.Bcast(y_primeg_flattend, root=0)
                g.comm.Bcast(y_goldg_flattend, root=0)
                y_primeg_flattend = np.split(y_primeg_flattend, num_pred)
                y_goldg_flattend = np.split(y_goldg_flattend, num_pred)
                y_primeg = [y.reshape((conf['model']['pred_batch_size'], max_length, 1)) for y in y_primeg_flattend]
                y_goldg = [y.reshape((conf['model']['pred_batch_size'], max_length, 1)) for y in y_goldg_flattend]
                y_primeg = np.concatenate(y_primeg, axis=0)
                y_goldg  = np.concatenate(y_goldg, axis=0)
                # Unpad each shot to its true length
                for idx, s in enumerate(shpzg):
                    trim = lambda nparry, s: nparry[0:int(s),:]
                    y_prime_global.append(trim(y_primeg[idx],s))
                    y_gold_global.append(trim(y_goldg[idx], s))

                disruptive_global += concatenate_sublists(
                    g.comm.allgather(disruptive))
                y_prime = []
                y_gold = []
                disruptive = []
                color = 2
                temp_predictor_only_comm.Free()

        y_prime_global = y_prime_global[:len(shot_list)]
        y_gold_global = y_gold_global[:len(shot_list)]
        disruptive_global = disruptive_global[:len(shot_list)]
        return y_prime_global, y_gold_global, disruptive_global

    def make_predictions_and_evaluate(self, shot_list):
        conf = self.conf
        y_prime, y_gold, disruptive = self.make_predictions(shot_list)
        analyzer = PerformanceAnalyzer(conf=conf)
        roc_area = analyzer.get_roc_area(y_prime, y_gold, disruptive)
        shot_list.set_weights(analyzer.get_shot_difficulty(y_prime, y_gold, disruptive))
        loss = get_loss_from_list(y_prime, y_gold, conf['data']['target'])
        return roc_area, loss

class CustomAUC(tf.keras.metrics.AUC):

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.clip_by_value(y_true, 0, 1)
        y_pred = tf.math.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)

@profile
def run(config: None):

    clear_session()

    # config['fs_path'] = f"/dev/shm/{node}"
    conf = parameters(config)

    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint(conf)

    handler = DataHandler(conf)
    builder = ModelBuilder(conf)

    # load data
    (shot_list_train, shot_list_valid, shot_list_test) = guarantee_preprocessed(conf)

    loader = handler.build_loader()
    train_dataset = handler.load_dataset(shot_list_train, loader)
    valid_dataset = handler.load_dataset(shot_list_valid, loader)

    # build model
    model = builder.build_model()
    model.summary()

    loss = conf['data']['target'].loss
    optimizer = builder.build_optimizer()
    model.compile(optimizer=optimizer, loss=loss, metrics=[CustomAUC(name="auc")])

    steps_per_epoch = loader.get_steps_per_epoch(shot_list_train)
    validation_steps = loader.get_steps_per_epoch(shot_list_valid)
    
    # train it
    history = model.fit(
        train_dataset,
        batch_size=conf['training']['batch_size'],
        epochs=conf['training']['num_epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=validation_steps,
    )

    with open('/lus/grand/projects/datascience/jgouneau/deephyper/frnn/exp/outputs/stateless_model.json', 'w') as file:
        json.dump(history.history, file)

    # evaluate it
    evaluator = ModelEvaluator(model, loader, conf)
    train_roc, train_loss = evaluator.make_predictions_and_evaluate(shot_list_train)
    valid_roc, valid_loss = evaluator.make_predictions_and_evaluate(shot_list_valid)
    test_roc, test_loss = evaluator.make_predictions_and_evaluate(shot_list_test)

    # print results
    print('======== RESULTS =======')
    print('Train Loss: {:.3e}'.format(train_loss))
    print('Train ROC: {:.4f}'.format(train_roc))
    print('Valid Loss: {:.3e}'.format(valid_loss))
    print('Valid ROC: {:.4f}'.format(valid_roc))
    print('Test Loss: {:.3e}'.format(test_loss))
    print('Test ROC: {:.4f}'.format(test_roc))

    # return valid_roc

if __name__ == '__main__':
    conf = {
        'pred_batch_size': 128,
        'length': 128,
        'rnn_size': 200,
        'rnn_layers': 2,
        'num_conv_filters': 128,
        'num_conv_layers': 3,
        'size_conv_filters': 3,
        'pool_size': 2,
        'dense_size': 128,
        'regularization': 0.001,
        'dense_regularization': 0.001,
        'lr': 2e-5,
        'lr_decay': 0.97,
        'momentum': 0.9,
        'dropout_prob': 0.1,
        'batch_size': 128,
    }
    
    t1 = time.time()
    run(conf)
    print("Run duration : {:.4f}".format(time.time() - t1))