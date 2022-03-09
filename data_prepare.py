import os
import yaml
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from data.vocabs import GRAPHEMES, PHONEMES

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_dataset(path_to_dict, batch_size=16, max_len_sequence=0):
    """
    Сырые данные преобразует в tf.dataset для обучения модели
    :param max_len_sequence:
    :param path_to_dict: путь к словарю grapheme to phoneme
    :return:tf.dataset
    """
    
    
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
#     tf.config.set_visible_devices(gpus[1], 'GPU')
    print('get_x_y...')
    words, spells = get_words_and_spells(path_to_dict, max_len_sequence)
    x, y = get_x_y(words, spells, GRAPHEMES, PHONEMES)
    del words
    del spells
    xy_sorted = sorted(zip(x, y), key=lambda a: len(a[0]))
    del x
    del y
    print('get_batches...')
    dataset = get_batches(xy_sorted, batch_size)
    return dataset


def get_words_and_spells(path_to_dict, max_len_sequence=0):
    """

    :param path_to_dict: путь к словарю grapheme to phoneme
    :param max_len_sequence:
    :return: words (список слов разбитых по графемам), spells(список произношений разбитых по фонемам)
    """
    with open(path_to_dict, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if '(' not in line]

    big = 0
    small = 0
    words = []
    spells = []
    print(len(lines))
    for line in tqdm(lines):
        chunks = line.split()
        graphs = list(chunks[0])
        phonemes = chunks[1:]
        if max_len_sequence != 0:
            if len(graphs) > max_len_sequence or len(phonemes) > max_len_sequence:
                continue
        if len(graphs) > len(phonemes):
            big += 1
        elif len(graphs) < len(phonemes):
            small += 1
        words.append(graphs)
        spells.append(phonemes)
    print(big, small)
    return words, spells


def get_x_y(words, spells, tokens_grapheme, tokens_phoneme):
    """
    векторизация входных и выходных последовательностей
    spell - последовательность фонем
    word - последовательность графем
    """
    words_one_hots = []
    for word in words:
        words_one_hots.append(get_sequence_one_hots(word, tokens_grapheme))
    spells_one_hots = []
    for spell in spells:
        spells_one_hots.append(get_sequence_one_hots(spell, tokens_phoneme))
    return words_one_hots, spells_one_hots


def get_sequence_one_hots(sequence, tokens):
    sequence_one_hots = []
    for element in sequence:
        sequence_one_hots.append(one_hot_encoding(element, tokens))
    return sequence_one_hots


def one_hot_encoding(element, tokens):
    vector = [0] * len(tokens)
    vector[tokens.index(element)] = 1
    return vector


def convert_list_to_np_array(x, y):
    x = [np.array(matrix) for matrix in x]
    x = [np.expand_dims(matrix, axis=0) for matrix in x]
    x = np.row_stack(x)
    y = [np.array(matrix) for matrix in y]
    y = [np.expand_dims(matrix, axis=0) for matrix in y]
    y = np.row_stack(y)
    return x, y


def get_batches(xy_sorted, batch_size):
    def gen():
        for x, y in xy_sorted:
            tensor_x = tf.constant(x)
            tensor_y = tf.constant(y)
            yield tensor_x, tensor_y

    def gen1():
        for x_batch, y_batch in zip(x_batches, y_batches):
            yield x_batch, y_batch
    with tf.device('/GPU:1'):
        dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32))
    # укладываем по батчам, padd-ит последовательночть в указанном shape None, так как в пакете тензоры должны быть одинакового размера
    m = dataset.padded_batch(batch_size, padded_shapes=([None, 35], [None, 49]), drop_remainder=True)
    del dataset
    print('padded')
    batches = []
    for batch in tqdm(m, total=int(len(xy_sorted) / batch_size)):
        batches.append(batch)
    del xy_sorted
    x_batches = []
    y_batches = []
    for x_batch, y_batch in tqdm(batches):
        x_len = x_batch.shape[1]
        y_len = y_batch.shape[1]
        # padd-им входную и выходную последовательность так как модель работает для одинаковой длины входа и выхода
        if x_len < y_len:
            x_batch = tf.pad(x_batch, paddings=[[0, 0], [0, y_len - x_len], [0, 0]])
        elif y_len < x_len:
            y_batch = tf.pad(y_batch, paddings=[[0, 0], [0, x_len - y_len], [0, 0]])
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    len_batches = len(batches)
    del batches
    #заменить None на batch_size
    dataset = tf.data.Dataset.from_generator(gen1, output_signature=(
        tf.TensorSpec(shape=(batch_size, None, 35), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, None, 49), dtype=tf.float32)
    )
                                             )
    train_ds, val_ds = get_dataset_partitions_tf(dataset, len_batches, shuffle_size=len_batches)
    return train_ds, val_ds


def get_dataset_partitions_tf(ds, ds_size, train_split=0.9, val_split=0.1, shuffle=True,
                              shuffle_size=10000):
    assert (train_split + val_split) == 1

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    return train_ds, val_ds


if __name__ == '__main__':
    with open('/home/murad/dss/g2p_github/configs/train_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    train_ds, val_ds = get_dataset(config['path_to_train'], batch_size=config['batch_size'])
    for batch in train_ds:
        print(batch)
        break
    tf.data.experimental.save(train_ds, config['path_train_dataset'])
    tf.data.experimental.save(val_ds, config['path_val_dataset'])
    # train_ds = tf.data.experimental.load(config['path_train_dataset'])
    # val_ds = tf.data.experimental.load(config['path_val_dataset'])
    print('ok')
