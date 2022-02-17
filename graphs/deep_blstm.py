from tensorflow.keras.layers import Bidirectional, LSTM, Input, Dense, Embedding, Flatten
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras import Model
import tensorflow as tf


class MyModel(Model):
    def __init__(self, len_phonemes, len_graphemes, rnn_units):
        super().__init__(self)
        self.lstm_1 = LSTM(rnn_units, return_sequences=True, input_shape=(None, len_graphemes))
        self.blstm = Bidirectional(LSTM(int(rnn_units/2), return_sequences=True), input_shape=(None, len_graphemes))
        self.lstm_2 = LSTM(128, return_sequences=True, input_shape=(None, 512))
        self.dense = Dense(len_phonemes, activation='softmax')

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        # x1 = self.lstm_1(x, training=training)
        # x2 = self.blstm(x, training=training)
        # x = tf.add(x1, x2)
        # x = self.lstm_2(x, training=training)
        x = self.dense(x, training=training)
        return x


def create_blstm(config_model):
    # inputs = Input((None, config_model['len_tokens_graphemes']))
    # blstm = Bidirectional(LSTM(config_model['units_lstm'], return_sequences=True), input_shape=(None, config_model['len_tokens_graphemes']))
    # x = blstm(inputs)
    # dense = Dense(config_model['output_dense_layer_neurons'], activation='softmax',
    #               kernel_initializer=RandomNormal(stddev=0.01, seed=42),
    #               bias_initializer=Zeros())
    # outputs = dense(x)
    # model = Model(inputs=inputs, outputs=outputs)
    model = MyModel(config_model['output_dense_layer_neurons'],
                    config_model['len_tokens_graphemes'],
                    config_model['units_lstm'])
    return model
