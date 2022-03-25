import tensorflow as tf

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

import numpy as np
from functools import partial

# from plasma.models.tcn import TCN
from plasma.preprocessor.preprocess import guarantee_preprocessed
from plasma.models.loader import Loader


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

    def build_model(self, custom_batch_size=None):
        conf = self.conf
        model_conf = conf['model']
        rnn_size = model_conf['rnn_size']
        rnn_type = model_conf['rnn_type']
        regularization = model_conf['regularization']
        dense_regularization = model_conf['dense_regularization']
        use_batch_norm = model_conf.get('use_batch_norm', False)

        dropout_prob = model_conf['dropout_prob']
        length = model_conf['length']
        pred_length = model_conf['pred_length']
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

        batch_size = model_conf['batch_size']

        if custom_batch_size is not None:
            batch_size = custom_batch_size

        if rnn_type == 'LSTM':
            rnn_model = LSTM
        elif rnn_type == 'CuDNNLSTM':
            rnn_model = CuDNNLSTM
        elif rnn_type == 'SimpleRNN':
            rnn_model = SimpleRNN
        else:
            print('Unkown Model Type, exiting.')
            exit(1)

        batch_input_shape = (batch_size, length, num_signals)

        indices_0d, indices_1d, num_0D, num_1D = self.get_0D_1D_indices()

        # PRE_RNN
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
        
        #
        x_input = Input(batch_shape=batch_input_shape)
        if num_1D > 0 or model_conf.get('extra_dense_input', False):
            x_in = TimeDistributed(pre_rnn_model)(x_input)
        else:
            x_in = x_input

        # ==========
        # TCN MODEL
        # ==========
        if model_conf.get('keras_tcn', False):
            tcn_layers = model_conf['tcn_layers']
            tcn_dropout = model_conf['tcn_dropout']
            nb_filters = model_conf['tcn_hidden']
            kernel_size = model_conf['kernel_size_temporal']
            nb_stacks = model_conf['tcn_nbstacks']
            use_skip_connections = model_conf['tcn_skip_connect']
            activation = model_conf['tcn_activation']
            use_batch_norm = model_conf['tcn_batch_norm']
            # for _ in range(model_conf['tcn_pack_layers']):
            #     x_in = TCN(
            #         use_batch_norm=use_batch_norm, activation=activation,
            #         use_skip_connections=use_skip_connections,
            #         nb_stacks=nb_stacks, kernel_size=kernel_size,
            #         nb_filters=nb_filters, num_layers=tcn_layers,
            #         dropout_rate=tcn_dropout)(x_in)
            #     x_in = Dropout(dropout_prob)(x_in)
        else:  # end TCN model
            # ==========
            # RNN MODEL
            # ==========
            # LSTM in ONNX: "The maximum opset needed by this model is only 9."
            model_kwargs = dict(return_sequences=return_sequences,
                                # batch_input_shape=batch_input_shape,
                                stateful=stateful,
                                kernel_regularizer=l2(regularization),
                                recurrent_regularizer=l2(regularization),
                                bias_regularizer=l2(regularization),
                                )
            if rnn_type != 'CuDNNLSTM':
                # Dropout (on linear transformation of recurrent state) is unsupported
                # in cuDNN library
                model_kwargs['recurrent_dropout'] = dropout_prob  # recurrent states
            model_kwargs['dropout'] = dropout_prob  # input states
            for _ in range(model_conf['rnn_layers']):
                x_in = rnn_model(rnn_size, **model_kwargs)(x_in)
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


def run(config: None):

    conf = parameters(config)

    if conf['data']['normalizer'] == 'minmax':
        from plasma.preprocessor.normalize import MinMaxNormalizer as Normalizer
    elif conf['data']['normalizer'] == 'meanvar':
        from plasma.preprocessor.normalize import MeanVarNormalizer as Normalizer
    elif conf['data']['normalizer'] == 'averagevar':
        from plasma.preprocessor.normalize import AveragingVarNormalizer as Normalizer
    else: # conf['data']['normalizer'] == 'var':
        from plasma.preprocessor.normalize import VarNormalizer as Normalizer


    normalizer = Normalizer(conf)
    (shot_list_train, shot_list_valid, shot_list_test) = guarantee_preprocessed(conf)
    normalizer.train()
    loader = Loader(conf, normalizer)

    train_batch_generator = partial(loader.training_batch_generator_partial_reset, shot_list=shot_list_train)
    valid_batch_generator = partial(loader.training_batch_generator_partial_reset, shot_list=shot_list_valid)

    train_dataset = tf.data.Dataset.from_generator(
        train_batch_generator,
        # output_signature=(
        #     tf.TensorSpec(shape=(None, length, num_signals), dtype=tf.float32),
        #     tf.RaggedTensorSpec(shape=(128, 128, 1), dtype=tf.float32)
        # )
    )

    valid_dataset = tf.data.Dataset.from_generator(valid_batch_generator)

    lr = conf['model']["lr"]
    decay = conf['model']["lr_decay"]
    momentum = conf['model']["momentum"]
    clipnorm = conf['model']["clipnorm"]
    optimizers = {
        "sgd": SGD,
        "momentum_sgd": SGD,
        "tf_momentum_sgd": tf.train.MomentumOptimizer,
        "adam": Adam,
        "tf_adam": tf.train.AdamOptimizer,
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
        "tf_momentum_sgd": {
            "learning_rate": lr,
            "momentum": momentum,
        },
        "adam": {
            "learning_rate": lr,
            "clipnorm": clipnorm,
        },
        "tf_adam": {
            "learning_rate": lr,
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
    
    opt_kwargs = optimizers_kwargs.get(conf['model']['optimizer'], optimizers_kwargs['Adam'])
    optimizer = optimizers.get(conf['model']['optimizer'], optimizers['Adam'])(**opt_kwargs)

    loss = conf['data']['target'].loss

    builder = ModelBuilder(conf)
    model = builder.build_model()
    model.summary()

    model.compile(optimizer=optimizer, loss=loss)

    model.fit(
        train_dataset,
        batch_size=conf['training']['batch_size'],
        epochs=conf['training']['num_epochs'],
        validation_data=valid_dataset,
    )




if __name__ == '__main__':
    conf = {
        'pred_length': 200,
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
        'mometum': 0.9,
        'dropout_prob': 0.1,
        'batch_size': 128,
    }
    
    run(conf)

    print("DONE")