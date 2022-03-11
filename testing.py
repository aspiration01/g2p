import yaml
import tensorflow as tf
import numpy as np
import editdistance
from tqdm import tqdm
from data_prepare import get_words_and_spells, get_x_y
from data.vocabs import GRAPHEMES, PHONEMES
from graphs.deep_blstm import create_blstm


def prepare_test_data(path_to_test_data):
    words, spells = get_words_and_spells(path_to_test_data)
    X, Y = get_x_y(words, spells, GRAPHEMES, PHONEMES)
    return X, Y


def func_testing(model, X, Y, result, PHONEMES):
    """
    вычисляем метрики wer(word error rate) = 1 - acc, cer(character error rate)
    avg_cer - средняя арифметичская(если мы хотим сделать акцент на слове)
    avg_cer_1 - cer посчитанный будто на одну длинную последо-сть.
    """
    
    total_cer = 0
    total_size = 0
    total_distance = 0
    nbr_right_answer = 0
    big = 0
    small = 0
    for x, y in tqdm(zip(X, Y), total=len(Y)):
        x = np.array(x, dtype=np.float32)
        z = np.expand_dims(x, axis=0)
        t = [x]
        y = np.array(y, dtype=np.float32)
        y_pred = model.predict(z)
        sequence_pred = [phoneme_distrib.argmax() for phoneme_distrib in y_pred[0]]
        sequence_true = [phoneme_distrib.argmax() for phoneme_distrib in np.array(y)]
        # print(len(sequence_true), len(sequence_pred))
        if len(sequence_true) > len(sequence_pred):
            small += 1
        elif len(sequence_true) < len(sequence_pred):
            big += 1
        distance = editdistance.eval(sequence_true, sequence_pred)
        if distance == 0:
            nbr_right_answer += 1
        cer = distance / len(sequence_true)
        total_cer += cer
        total_distance += distance
        total_size += len(sequence_true)
        phonemes_true = decoding_phonemes(sequence_true, PHONEMES)
        phonemes_pred = decoding_phonemes(sequence_pred, PHONEMES)
        result.write(f'{phonemes_true}\t{phonemes_pred}\t{cer}\n')
    avg_cer = total_cer / len(Y)
    avg_cer_1 = total_distance / total_size
    wer = 1 - nbr_right_answer / len(Y)
    result.write(f'avg_cer: {avg_cer_1}\t wer: {wer}')
    print(small, big)
    print(f'average_cer={round(avg_cer_1, 2)}')
    print(f'wer={round(wer, 2)}')
    print(f'accuracy={round(nbr_right_answer/len(Y), 2)}')
    
    
def decoding_phonemes(indicies, tokens_phoneme):
    phonemes = []
    for index in indicies:
        phonemes.append(tokens_phoneme[index])
    return ' '.join(phonemes)


if __name__ == '__main__':
    with open('configs/test_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open('models/deep_blstm/config_deep_blstm.yaml', 'r') as f:
        config_model = yaml.load(f, Loader=yaml.FullLoader)
    model = create_blstm(config_model)
    model.load_weights(config['path_checkpoint'])
    result = open(config['path_to_result_model'], 'w')
    X, Y = prepare_test_data(config['path_test_data'])
    X = X[:100]
    Y = Y[:100]
    func_testing(model, X, Y, result, PHONEMES)
    result.close()