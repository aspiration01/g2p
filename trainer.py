import yaml
import tensorflow as tf
from graphs.simple_rnn import create_simple_rnn
from graphs.lstm import create_lstm
from graphs.deep_blstm import create_blstm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

optimizer = Adam(learning_rate=0.001)
loss_fn = CategoricalCrossentropy()


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_batch_pred = model(x, training=True)
        loss_value = loss_fn(y, y_batch_pred)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


def train_model(model, train_dataset, val_dataset, epochs, batch_size, checkpoint_filepath):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        sum_loss = 0.0
        count = 0.0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train)
            sum_loss += loss_value
            count += 1
            if step % 200 == 0:
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
        print("epoch=%d val_loss=%.4f" % (epoch, float(val_avg_loss)))
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
    #                                                                save_weights_only=True,
    #                                                                monitor='val_loss',
    #                                                                mode='min',
    #                                                                save_best_only=True)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # model.fit(train_dataset, epochs=1, verbose=1, validation_data=val_dataset,
    #           callbacks=[early_stopping, model_checkpoint_callback])


if __name__ == '__main__':
    with open('configs/train_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(config['path_to_config_model'], 'r') as f:
        config_model = yaml.load(f, Loader=yaml.FullLoader)
    print("\nLOAD DATA")
    train_dataset = tf.data.experimental.load(config['path_train_dataset'])
    val_dataset = tf.data.experimental.load(config['path_val_dataset'])
    train_dataset.take(1000)
    val_dataset.take(1000)
    batch_size = config['batch_size']
    epochs = config['epochs']
    # model = create_simple_rnn(config_model)
    # model = create_lstm(config_model)
    print("\nCREATE MODEL")
    model = create_blstm(config_model)
    for input_example_batch, target_example_batch in train_dataset.take(1):
        # print(input_example_batch)
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    model.summary()
    train_model(model, train_dataset, val_dataset, epochs, batch_size, config['path_to_checkpoint'])
    print("\nSAVING MODEL")
    print(f"   * load best checkpoint")
    model.load_weights(config['path_to_checkpoint'])
    print(f"   * export to: {config['path_to_model']}")
    model.save(config['path_to_model'], save_format="tf")
    print(f"   * done!")
    print("\nFINISHED!")
