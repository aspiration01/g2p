import tensorflow as tf


def create_lstm(config_model):
    inputs = tf.keras.layers.Input((None, config_model['len_tokens_graphemes']))
    lstm = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True,
                               input_shape=(None, config_model['len_tokens_graphemes']))
    x = lstm(inputs)
    dense = tf.keras.layers.Dense(49, activation='softmax',
                                  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=42),
                                  bias_initializer=tf.keras.initializers.Zeros())
    outputs = dense(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model