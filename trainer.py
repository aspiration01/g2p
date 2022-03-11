import yaml
import time
import tensorflow as tf
import numpy as np
from graphs.simple_rnn import create_simple_rnn
from graphs.lstm import create_lstm
from graphs.deep_blstm import create_blstm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


from testing import prepare_test_data, func_testing
from data.vocabs import PHONEMES

optimizer = Adam(learning_rate=0.001)
loss_fn = CategoricalCrossentropy()


def train_step(x, y):
    with tf.GradientTape() as tape:
        y_batch_pred = model(x, training=True)
        loss_value = loss_fn(y, y_batch_pred)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


def train_model(model, train_dataset, val_dataset, epochs, batch_size, checkpoint_filepath, test_config):
    sum_loss_epoch = 0
    count_epoch = 0
    best_avg_val_loss = 1.0
    result = open(test_config['path_to_result_model'], 'w')
    X, y = prepare_test_data(test_config['path_test_data'])
    for epoch in range(epochs):
        start_time = time.time()
        print("\nStart of epoch %d" % (epoch,))
        sum_loss = 0.0
        count = 0.0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # np_x = np.array(x_batch_train)
            # np_y = np.array(y_batch_train)
            loss_value = train_step(x_batch_train, y_batch_train)
            sum_loss += loss_value
            sum_loss_epoch += loss_value
            count += 1
            count_epoch += 1
            if step % 2000 == 0:
                avg_loss = sum_loss / count
                print(
                    "step=%d loss=%.4f"
                    % (step, float(avg_loss))
                )
                sum_loss = 0.0
                count = 0.0
        val_sum_loss = 0.0
        val_count = 0
        for x_batch_val, y_batch_val in val_dataset:
            y_batch_pred_val = model(x_batch_val, training=False)
            val_loss = loss_fn(y_batch_val, y_batch_pred_val)
            val_sum_loss += val_loss
            val_count += 1
        val_avg_loss = val_sum_loss / val_count
        avg_loss_epoch = sum_loss_epoch / count_epoch
        print(time.time())
        print("epoch=%d loss=%.4f val_loss=%.4f time_epoch=%.1f" % (epoch, avg_loss_epoch, 
                                                                    val_avg_loss, 
                                                                    time.time() - start_time))
        best_avg_val_loss = callback_checkpoint(model, val_avg_loss, best_avg_val_loss, checkpoint_filepath)
        func_testing(model, X, y, result, PHONEMES)


def callback_checkpoint(model, val_loss, best_val_loss, checkpoint_filepath):
    if val_loss < best_val_loss:
        model.save_weights(checkpoint_filepath)
        return val_loss
    else:
        return best_val_loss


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) >= 2:
        tf.config.set_visible_devices(gpus[1], 'GPU')
    elif len(gpus) == 1:
        tf.config.set_visible_devices(gpus[0], 'GPU')
    with open('configs/train_config.yaml', 'r') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    with open('configs/test_config.yaml', 'r') as f:
        test_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(train_config['path_to_config_model'], 'r') as f:
        config_model = yaml.load(f, Loader=yaml.FullLoader)
    print("\nLOAD DATA")
    train_dataset = tf.data.experimental.load(train_config['path_train_dataset'])
    val_dataset = tf.data.experimental.load(train_config['path_val_dataset'])
# #     train_dataset.take(1000)
# #     val_dataset.take(1000)
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    # model = create_simple_rnn(config_model)
    # model = create_lstm(config_model)
    print("\nCREATE MODEL")
    model = create_blstm(config_model)
    for input_example_batch, target_example_batch in train_dataset.take(1):
        # print(input_example_batch)
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    model.summary()
    train_model(model, train_dataset, val_dataset, epochs, batch_size, train_config['path_to_checkpoint'], test_config)
    print("\nSAVING MODEL")
    print(f"   * load best checkpoint")
    model.load_weights(train_config['path_to_checkpoint'])
    print(f"   * export to: {train_config['path_to_save_model']}")
    model.save(train_config['path_to_save_model'], save_format="tf")
    print(f"   * done!")
    print("\nFINISHED!")
